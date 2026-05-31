.PHONY: all analysis paper clean

all: analysis paper

analysis:
	python3 scripts/build_paper_artifacts.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

clean:
	cd paper && latexmk -C main.tex
