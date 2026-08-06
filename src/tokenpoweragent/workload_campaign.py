"""Pre-registered workload-transfer prediction and measurement campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tokenpoweragent.evidence import EvidenceRecord, EvidenceStatus, EvidenceStore
from tokenpoweragent.executors.serving import (
    ServerEnvironment,
    ServingSandboxExecutor,
)
from tokenpoweragent.executors.topology import TopologySandboxExecutor
from tokenpoweragent.schema import Candidate, EvidenceLevel
from tokenpoweragent.twin.topology import (
    CalibrationError,
    CalibrationProfile,
    InferenceWorkload,
    ProjectionBackend,
    ServingConfiguration,
)


class WorkloadCampaignError(ValueError):
    """Raised when a workload campaign violates its frozen contract."""


_DATASET_SPLITS = {"calibration", "validation", "holdout", "diagnostic"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkloadPoint:
    point_id: str
    dataset_split: str
    workload: InferenceWorkload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WorkloadPoint":
        point_id = str(raw.get("id", "")).strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", point_id):
            raise WorkloadCampaignError("workload requires a stable lowercase id")
        dataset_split = str(raw.get("dataset_split", "")).strip().lower()
        if dataset_split not in _DATASET_SPLITS:
            raise WorkloadCampaignError(
                "workload %s has an invalid dataset_split" % point_id
            )
        try:
            workload = InferenceWorkload.from_mapping(raw)
        except CalibrationError as exc:
            raise WorkloadCampaignError(
                "workload %s is invalid: %s" % (point_id, exc)
            ) from exc
        if workload.num_requests < workload.concurrency:
            raise WorkloadCampaignError(
                "workload %s needs at least one request per concurrent slot"
                % point_id
            )
        if workload.num_requests % workload.concurrency:
            raise WorkloadCampaignError(
                "workload %s num_requests must be divisible by concurrency"
                % point_id
            )
        return cls(point_id, dataset_split, workload)


@dataclass(frozen=True)
class WorkloadCampaign:
    schema_version: str
    campaign_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    max_model_len: int
    configuration: Mapping[str, Any]
    power_limit_w: float
    repeats: int
    num_warmups: int
    sample_ms: int
    schedule: str
    workloads: Tuple[WorkloadPoint, ...]

    @classmethod
    def load(cls, path: Path) -> "WorkloadCampaign":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkloadCampaignError("cannot read campaign JSON: %s" % exc) from exc
        if not isinstance(raw, Mapping):
            raise WorkloadCampaignError("campaign root must be an object")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "WorkloadCampaign":
        campaign_id = str(raw.get("campaign_id", "")).strip()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", campaign_id):
            raise WorkloadCampaignError("campaign_id must be stable and lowercase")
        model = raw.get("model")
        measurement = raw.get("measurement")
        configuration = raw.get("serving_configuration")
        workloads_raw = raw.get("workloads")
        if not isinstance(model, Mapping):
            raise WorkloadCampaignError("campaign.model must be an object")
        if not isinstance(measurement, Mapping):
            raise WorkloadCampaignError("campaign.measurement must be an object")
        if not isinstance(configuration, Mapping):
            raise WorkloadCampaignError(
                "campaign.serving_configuration must be an object"
            )
        if not isinstance(workloads_raw, list) or not workloads_raw:
            raise WorkloadCampaignError("campaign requires non-empty workloads")

        workloads = tuple(WorkloadPoint.from_mapping(item) for item in workloads_raw)
        ids = [point.point_id for point in workloads]
        if len(ids) != len(set(ids)):
            raise WorkloadCampaignError("campaign workload ids must be unique")

        model_id = str(model.get("id", "")).strip()
        model_revision = str(model.get("revision", "")).strip()
        tokenizer_revision = str(model.get("tokenizer_revision", "")).strip()
        if not model_id or not model_revision or not tokenizer_revision:
            raise WorkloadCampaignError("campaign must pin model and tokenizer revisions")
        max_model_len = _positive_int(model, "max_model_len")
        for point in workloads:
            if point.workload.input_tokens + point.workload.output_tokens > max_model_len:
                raise WorkloadCampaignError(
                    "workload %s exceeds max_model_len" % point.point_id
                )

        repeats = _positive_int(measurement, "repeats")
        num_warmups = _nonnegative_int(measurement, "num_warmups")
        sample_ms = _positive_int(measurement, "sample_ms")
        try:
            power_limit_w = float(measurement["power_limit_w"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkloadCampaignError("measurement.power_limit_w must be numeric") from exc
        if not math.isfinite(power_limit_w) or power_limit_w <= 0:
            raise WorkloadCampaignError("measurement.power_limit_w must be positive")
        schedule = str(measurement.get("schedule", "")).strip().lower()
        if schedule != "cyclic-balanced":
            raise WorkloadCampaignError(
                "measurement.schedule must be cyclic-balanced"
            )

        if configuration.get("prefix_caching") is not False:
            raise WorkloadCampaignError(
                "serving_configuration must explicitly disable prefix caching"
            )
        try:
            serving_config = ServingConfiguration.from_mapping(
                configuration, workloads[0].workload
            )
        except CalibrationError as exc:
            raise WorkloadCampaignError("invalid serving configuration: %s" % exc) from exc
        if serving_config.required_gpus != 1:
            raise WorkloadCampaignError(
                "this serving campaign supports exactly one GPU"
            )
        if serving_config.max_num_seqs < max(
            point.workload.concurrency for point in workloads
        ):
            raise WorkloadCampaignError(
                "max_num_seqs is smaller than a campaign concurrency"
            )
        normalized_config = serving_config.to_dict()
        normalized_config["prefix_caching"] = False

        return cls(
            schema_version=str(raw.get("schema_version", "1.0")),
            campaign_id=campaign_id,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            max_model_len=max_model_len,
            configuration=normalized_config,
            power_limit_w=power_limit_w,
            repeats=repeats,
            num_warmups=num_warmups,
            sample_ms=sample_ms,
            schedule=schedule,
            workloads=workloads,
        )

    def candidate(self, point: WorkloadPoint) -> Candidate:
        return Candidate(
            candidate_id=point.point_id,
            config=dict(self.configuration),
            required_gpus=1,
            target_nodes=1,
        )

    def assert_profile(self, profile: CalibrationProfile) -> None:
        if profile.model.model_id != self.model_id:
            raise WorkloadCampaignError("profile model does not match campaign")
        if profile.model.revision != self.model_revision:
            raise WorkloadCampaignError("profile model revision does not match campaign")

    def assert_server_environment(self, environment: ServerEnvironment) -> None:
        actual = environment.configuration
        for key, expected in self.configuration.items():
            if actual.get(key) != expected:
                raise WorkloadCampaignError(
                    "server configuration differs at %s: expected %r, found %r"
                    % (key, expected, actual.get(key))
                )
        if actual.get("model_revision") != self.model_revision:
            raise WorkloadCampaignError("server model revision does not match campaign")
        if actual.get("tokenizer_revision") != self.tokenizer_revision:
            raise WorkloadCampaignError(
                "server tokenizer revision does not match campaign"
            )

    def assert_measurement_record(
        self,
        point: WorkloadPoint,
        record: EvidenceRecord,
        server_image_id: str,
        client_image_id: Optional[str] = None,
    ) -> None:
        provenance = record.provenance
        if provenance.get("dataset_split") != point.dataset_split:
            raise WorkloadCampaignError("measurement dataset split drifted")
        if provenance.get("model") != self.model_id:
            raise WorkloadCampaignError("measurement model drifted")
        if provenance.get("server_image_id") != server_image_id:
            raise WorkloadCampaignError("measurement server image drifted")
        if (
            client_image_id is not None
            and provenance.get("client_image_id") != client_image_id
        ):
            raise WorkloadCampaignError("measurement client image drifted")
        try:
            power_limit_w = float(provenance["power_limit_readback_w"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkloadCampaignError(
                "measurement lacks a valid power limit"
            ) from exc
        if (
            not math.isfinite(power_limit_w)
            or abs(power_limit_w - self.power_limit_w) > 1.0
        ):
            raise WorkloadCampaignError("measurement power limit drifted")
        configuration = provenance.get("server_configuration")
        if not isinstance(configuration, Mapping):
            raise WorkloadCampaignError("measurement lacks server configuration")
        self.assert_server_environment(
            ServerEnvironment(
                image=str(provenance.get("server_image", "")),
                image_id=server_image_id,
                command=tuple(provenance.get("server_command", ())),
                configuration=configuration,
            )
        )
        workload = provenance.get("workload")
        expected_workload = {
            "input_len": point.workload.input_tokens,
            "output_len": point.workload.output_tokens,
            "num_prompts": point.workload.num_requests,
            "num_warmups": self.num_warmups,
            "request_rate": (
                "inf"
                if point.workload.request_rate_req_s is None
                else "%g" % point.workload.request_rate_req_s
            ),
            "max_concurrency": point.workload.concurrency,
            "temperature": 0,
            "ignore_eos": True,
        }
        if workload != expected_workload:
            raise WorkloadCampaignError("measurement workload drifted")


def freeze_workload_campaign(
    campaign_path: Path,
    calibration_path: Path,
    output_path: Path,
    summary_path: Path,
    manifest_path: Path,
    backend: ProjectionBackend = ProjectionBackend.L0_A,
) -> Mapping[str, Any]:
    try:
        backend = ProjectionBackend.parse(backend)
    except CalibrationError as exc:
        raise WorkloadCampaignError(str(exc)) from exc
    for path in (output_path, summary_path, manifest_path):
        if Path(path).exists():
            raise WorkloadCampaignError(
                "refusing to overwrite frozen artifact: %s" % path
            )

    campaign = WorkloadCampaign.load(campaign_path)
    try:
        profile = CalibrationProfile.load(calibration_path)
    except CalibrationError as exc:
        raise WorkloadCampaignError("invalid calibration profile: %s" % exc) from exc
    campaign.assert_profile(profile)
    campaign_hash = sha256_file(campaign_path)
    profile_hash = sha256_file(calibration_path)
    records = EvidenceStore()
    for point in campaign.workloads:
        executor = TopologySandboxExecutor(
            profile,
            point.workload,
            expected_model=campaign.model_id,
            profile_path=calibration_path,
            scenario_path=campaign_path,
            backend=backend,
        )
        record = executor.execute(
            campaign.candidate(point), EvidenceLevel.L0, seed=0
        )
        if record.status != EvidenceStatus.SUCCEEDED:
            raise WorkloadCampaignError(
                "cannot freeze failed prediction for %s: %s"
                % (point.point_id, record.failure_reason or "unknown failure")
            )
        provenance = dict(record.provenance)
        provenance.update(
            {
                "campaign_id": campaign.campaign_id,
                "campaign_sha256": campaign_hash,
                "workload_id": point.point_id,
                "dataset_split": point.dataset_split,
            }
        )
        records.append(replace(record, provenance=provenance))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    records.write_jsonl(output_path)
    summary = {
        "schema_version": campaign.schema_version,
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign_hash,
        "profile_id": profile.profile_id,
        "profile_sha256": profile_hash,
        "profile_publication_eligible": profile.publication_eligible,
        "uncertainty_calibrated": profile.uncertainty_calibrated,
        "level": EvidenceLevel.L0.name,
        "sandbox_backend": backend.display_name,
        "projection_backend": backend.value,
        "prediction_count": len(records.records),
        "workload_ids": [point.point_id for point in campaign.workloads],
        "dataset_splits": {
            point.point_id: point.dataset_split for point in campaign.workloads
        },
        "validation_required": True,
    }
    Path(summary_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prediction_hash = sha256_file(output_path)
    summary_hash = sha256_file(summary_path)
    Path(manifest_path).write_text(
        "".join(
            "%s  %s\n" % (digest, path)
            for digest, path in (
                (campaign_hash, campaign_path),
                (profile_hash, calibration_path),
                (prediction_hash, output_path),
                (summary_hash, summary_path),
            )
        ),
        encoding="utf-8",
    )
    return summary


def verify_frozen_campaign(
    campaign_path: Path,
    predictions_path: Path,
    manifest_path: Path,
) -> Tuple[str, str]:
    entries = _read_manifest(manifest_path)
    campaign_hash = _verify_manifest_path(entries, campaign_path)
    prediction_hash = _verify_manifest_path(entries, predictions_path)
    campaign = WorkloadCampaign.load(campaign_path)
    records = EvidenceStore.read_jsonl(predictions_path).records
    if len(records) != len(campaign.workloads):
        raise WorkloadCampaignError("frozen prediction count does not match campaign")
    expected = {point.point_id: point for point in campaign.workloads}
    if {record.candidate_id for record in records} != set(expected):
        raise WorkloadCampaignError("frozen prediction ids do not match campaign")
    for record in records:
        point = expected[record.candidate_id]
        if record.status != EvidenceStatus.SUCCEEDED:
            raise WorkloadCampaignError("frozen campaign contains a failed prediction")
        if record.provenance.get("campaign_sha256") != campaign_hash:
            raise WorkloadCampaignError("prediction campaign hash is inconsistent")
        if record.provenance.get("dataset_split") != point.dataset_split:
            raise WorkloadCampaignError("prediction dataset split is inconsistent")
        if record.provenance.get("workload_id") != point.point_id:
            raise WorkloadCampaignError("prediction workload id is inconsistent")
        decomposition = record.provenance.get("decomposition")
        if not isinstance(decomposition, Mapping):
            raise WorkloadCampaignError("prediction lacks decomposition")
        if decomposition.get("workload") != point.workload.to_dict():
            raise WorkloadCampaignError("prediction workload is inconsistent")
    return campaign_hash, prediction_hash


def campaign_schedule(
    workloads: Sequence[WorkloadPoint], repeats: int
) -> Tuple[Tuple[int, int, WorkloadPoint], ...]:
    if not workloads or repeats < 1:
        raise WorkloadCampaignError("schedule requires workloads and repeats")
    count = len(workloads)
    schedule = []
    for repeat_index in range(repeats):
        offset = (repeat_index * count) // repeats
        ordered = tuple(workloads[offset:]) + tuple(workloads[:offset])
        start = len(schedule)
        schedule.extend(
            (start + position, repeat_index, point)
            for position, point in enumerate(ordered)
        )
    return tuple(schedule)


def serving_candidate_config(
    campaign: WorkloadCampaign,
    point: WorkloadPoint,
    runtime: Mapping[str, Any],
    campaign_index: int,
) -> Mapping[str, Any]:
    rate = point.workload.request_rate_req_s
    return {
        **runtime,
        "model": campaign.model_id,
        "served_model_name": runtime["served_model_name"],
        "power_limit_w": campaign.power_limit_w,
        "campaign_index": campaign_index,
        "input_len": point.workload.input_tokens,
        "output_len": point.workload.output_tokens,
        "num_prompts": point.workload.num_requests,
        "num_warmups": campaign.num_warmups,
        "request_rate": "inf" if rate is None else "%g" % rate,
        "max_concurrency": point.workload.concurrency,
        "dataset_split": point.dataset_split,
    }


def annotate_measurement(
    record: EvidenceRecord,
    campaign: WorkloadCampaign,
    point: WorkloadPoint,
    repeat_index: int,
    campaign_index: int,
    campaign_hash: str,
    prediction_hash: str,
) -> EvidenceRecord:
    provenance = dict(record.provenance)
    provenance.update(
        {
            "campaign_id": campaign.campaign_id,
            "campaign_sha256": campaign_hash,
            "workload_id": point.point_id,
            "campaign_repeat": repeat_index,
            "campaign_schedule_index": campaign_index,
            "frozen_prediction_sha256": prediction_hash,
        }
    )
    return replace(record, provenance=provenance)


def run_serving_campaign(
    campaign_path: Path,
    predictions_path: Path,
    manifest_path: Path,
    output_path: Path,
    executor: ServingSandboxExecutor,
    runtime: Mapping[str, Any],
    resume: bool = False,
) -> Mapping[str, Any]:
    campaign = WorkloadCampaign.load(campaign_path)
    if executor.sample_ms != campaign.sample_ms:
        raise WorkloadCampaignError(
            "executor sample interval does not match campaign"
        )
    campaign_hash, prediction_hash = verify_frozen_campaign(
        campaign_path, predictions_path, manifest_path
    )
    schedule = campaign_schedule(campaign.workloads, campaign.repeats)
    first_index, _, first_point = schedule[0]
    first_candidate = Candidate(
        candidate_id=first_point.point_id,
        config=serving_candidate_config(
            campaign, first_point, runtime, first_index
        ),
    )
    server_environment = executor.inspect_environment(first_candidate)
    campaign.assert_server_environment(server_environment)

    output_path = Path(output_path)
    if output_path.exists() and not resume:
        raise WorkloadCampaignError(
            "measurement output exists; pass resume=True to continue"
        )
    records = (
        EvidenceStore.read_jsonl(output_path)
        if output_path.exists()
        else EvidenceStore()
    )
    completed: Dict[Tuple[str, int], EvidenceRecord] = {}
    point_map = {point.point_id: point for point in campaign.workloads}
    client_image_id = None
    for record in records.records:
        provenance = record.provenance
        if provenance.get("campaign_id") != campaign.campaign_id:
            raise WorkloadCampaignError(
                "existing measurement belongs to another campaign"
            )
        if provenance.get("campaign_sha256") != campaign_hash:
            raise WorkloadCampaignError("existing measurement campaign hash differs")
        if provenance.get("frozen_prediction_sha256") != prediction_hash:
            raise WorkloadCampaignError(
                "existing measurement prediction hash differs"
            )
        point_id = str(provenance.get("workload_id", ""))
        if point_id not in point_map:
            raise WorkloadCampaignError("existing measurement has unknown workload")
        try:
            repeat_index = int(provenance["campaign_repeat"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkloadCampaignError(
                "existing measurement has invalid campaign_repeat"
            ) from exc
        key = (point_id, repeat_index)
        if key in completed:
            raise WorkloadCampaignError("existing measurements contain duplicates")
        if record.status != EvidenceStatus.SUCCEEDED:
            raise WorkloadCampaignError(
                "cannot resume a campaign containing failed evidence"
            )
        campaign.assert_measurement_record(
            point_map[point_id],
            record,
            server_environment.image_id,
            client_image_id,
        )
        observed_client_image_id = record.provenance.get("client_image_id")
        if client_image_id is None and observed_client_image_id is not None:
            client_image_id = str(observed_client_image_id)
        completed[key] = record

    output_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    for campaign_index, repeat_index, point in schedule:
        key = (point.point_id, repeat_index)
        if key in completed:
            continue
        candidate = Candidate(
            candidate_id=point.point_id,
            config=serving_candidate_config(
                campaign, point, runtime, campaign_index
            ),
        )
        record = executor.execute(candidate, EvidenceLevel.L1, seed=repeat_index)
        campaign.assert_measurement_record(
            point,
            record,
            server_environment.image_id,
            client_image_id,
        )
        observed_client_image_id = record.provenance.get("client_image_id")
        if client_image_id is None and observed_client_image_id is not None:
            client_image_id = str(observed_client_image_id)
        record = annotate_measurement(
            record,
            campaign,
            point,
            repeat_index,
            campaign_index,
            campaign_hash,
            prediction_hash,
        )
        records.append(record)
        records.write_jsonl(output_path)
        added += 1
        if record.status != EvidenceStatus.SUCCEEDED:
            raise WorkloadCampaignError(
                "campaign stopped after failed workload %s repeat %d"
                % (point.point_id, repeat_index)
            )

    return {
        "campaign_id": campaign.campaign_id,
        "campaign_sha256": campaign_hash,
        "prediction_sha256": prediction_hash,
        "scheduled_measurements": len(schedule),
        "existing_measurements": len(completed),
        "added_measurements": added,
        "completed_measurements": len(records.records),
        "output": str(output_path),
    }


def _read_manifest(path: Path) -> Mapping[str, str]:
    entries: Dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
            raise WorkloadCampaignError("freeze manifest has an invalid line")
        listed = parts[1].lstrip("* ")
        if listed in entries:
            raise WorkloadCampaignError("freeze manifest has duplicate paths")
        entries[listed] = parts[0].lower()
    return entries


def _verify_manifest_path(entries: Mapping[str, str], path: Path) -> str:
    matches = [
        digest
        for listed, digest in entries.items()
        if listed == str(path) or Path(listed).name == Path(path).name
    ]
    if len(matches) != 1:
        raise WorkloadCampaignError(
            "freeze manifest must contain exactly one entry for %s" % path
        )
    actual = sha256_file(path)
    if matches[0] != actual:
        raise WorkloadCampaignError("frozen artifact hash mismatch: %s" % path)
    return actual


def _positive_int(raw: Mapping[str, Any], key: str) -> int:
    try:
        value = int(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkloadCampaignError("%s must be a positive integer" % key) from exc
    if value < 1:
        raise WorkloadCampaignError("%s must be a positive integer" % key)
    return value


def _nonnegative_int(raw: Mapping[str, Any], key: str) -> int:
    try:
        value = int(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkloadCampaignError("%s must be a non-negative integer" % key) from exc
    if value < 0:
        raise WorkloadCampaignError("%s must be a non-negative integer" % key)
    return value
