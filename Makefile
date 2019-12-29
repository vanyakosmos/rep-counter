run:
	python main.py -d 1 --bs 2 --device 0

mock:
	python main.py -d 33 --bs 2 --device -1 -s waves

test:
	python tests.py


.DEFAULT_GOAL := run
