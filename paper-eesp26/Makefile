.PHONY: pdf check clean

pdf:
	latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error -outdir=build main.tex

check: pdf
	python3 analysis/check_submission.py

clean:
	latexmk -C -outdir=build main.tex
