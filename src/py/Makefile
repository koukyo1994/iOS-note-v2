IMG := ios-note
TAG := trial

font-setup:
	./setup.sh

docker-build:
	make -C ../docker

env:
	docker run -it --rm --init \
	--ipc host \
	--name ios-note \
	--volume `pwd`:/content \
	-w /content \
	${IMG}:${TAG} /bin/bash

create-dataset-test:
	chmod +x setup.sh
	make font-setup
	python dataset.py
