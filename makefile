.PHONY: install run yolo_train cnn_train pipeline test clean

install:
	python -m pip install -e .

run:
	python -m card-detector

yolo_train:
	card-detector yolo-train

cnn_train:
	card-detector cnn-train

pipeline:
	card-detector --input $(INPUT) --output $(OUTPUT)

test:
	pytest -q --disable-warnings --maxfail=1

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info
