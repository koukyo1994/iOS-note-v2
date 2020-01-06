IMG := ios-note
TAG := trial
NSAMPLES := 1000

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

create-dataset:
	chmod +x setup.sh
	make font-setup
	python dataset.py --n_samples ${NSAMPLES}
