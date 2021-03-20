#################################
#  Singularity+Vagrant helpers  #
#################################

# This Makefile provides helper commands for orchestrating the Singularity
# container for this repository. It requires GNU Make (may be called gmake on
# your system).
CONTAINER = ml-wf-lab-container.sif

# Instance name to use when starting the container
INSTANCE_NAME = ml-wf

# On non-Linux (e.g. macOS) platforms, we will attempt to set up a Vagrant box
# to run the containers in (see accompanying Vagrantfile). If necessary, this
# behavior can be overridden by setting the USE_VAGRANT environment variable.
PLATFORM := $(shell uname)
ifneq "${PLATFORM}" "Linux"
USE_VAGRANT ?= 1
endif

# If it's set to 0, undefine it so that our conditionals below work properly.
ifeq "${USE_VAGRANT}" "0"
override undefine USE_VAGRANT
endif

# We bind the current directory over /app/src in the container by default.
SGTY_BIND_FLAGS ?= --bind .:/app/src

# Command paths. These can be overridden if your system gives them different
# names.
VAGRANT = vagrant
PYTHON3 = python3
SINGULARITY = singularity

# Magic help snippet from https://victoria.dev/blog/how-to-create-a-self-documenting-makefile/
# Looks for double-hash comments and displays them in a nice list.
.PHONY: help
help: ## Show this help
	@echo "Usage: make <target>"
	@echo
	@echo "Available targets:"
	@egrep -h '\s##\s' ${MAKEFILE_LIST} \
	 | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo

# Python script to launch Jupyter in the browser
BROWSER_SCRIPT:='import sys, json, webbrowser; d=json.load(sys.stdin); webbrowser.open("http://localhost:{}/?token={}".format(d["port"], d["token"]))'

.PHONY: build up down status shell tail browser
ifndef USE_VAGRANT

# These are the targets that actually do the work. If USE_VAGRANT is set, then
# they will be run inside the Vagrant box; otherwise, they are run directly on
# the host machine.

.PHONY: start

%.sif: %.def poetry.lock
	${MAKE} down
	${SINGULARITY} build --force --fakeroot $@ $<

build: ${CONTAINER} ## Build the container image

start: build
	${SINGULARITY} instance start ${SGTY_BIND_FLAGS} ${CONTAINER} ${INSTANCE_NAME}

up: start browser ## Start a containerized Jupyter Lab

down: ## Stop a running Jupyter Lab instance
	-${SINGULARITY} instance stop ${INSTANCE_NAME}

status: ## Check on the status of Jupyter Lab
	${SINGULARITY} exec ${CONTAINER} jupyter lab list

shell: build ## Open a shell inside a new container instance
	${SINGULARITY} shell ${SGTY_BIND_FLAGS} ${CONTAINER}

tail: ## Follow the Jupyter Lab instance logs
	@${SINGULARITY} instance list -j ${INSTANCE_NAME} \
	| ${PYTHON3} -c 'import sys, json; print("\0".join("{}\0{}".format(i["logOutPath"],i["logErrPath"]) for i in json.load(sys.stdin)["instances"]), end="")' \
	| xargs -0 tail -F

browser: ## Open the running Jupyter Lab instance in the browser
	cat "~/.local/share/jupyter/runtime/jpserver-*.json" \
	| ${PYTHON3} -c ${BROWSER_SCRIPT}

else # USE_VAGRANT defined

# If we're using vagrant, and we're on the host, these targets arrange to have
# the ones above run inside the guest.

.PHONY: vm-up halt

vm-up:
	${VAGRANT} up --no-provision

build: vm-up
	${VAGRANT} ssh -c 'make USE_VAGRANT=0 -C /vagrant build'

up: vm-up
	${VAGRANT} ssh -c 'make USE_VAGRANT=0 -C /vagrant start'
	${MAKE} browser

down:
	-${VAGRANT} ssh -c 'make USE_VAGRANT=0 -C /vagrant down'

halt: down ## Like 'down', but also shuts down the VM
	-${VAGRANT} halt

status:
	@${VAGRANT} ssh -c 'make USE_VAGRANT=0 -s -C /vagrant status'

shell: vm-up
	${VAGRANT} ssh -c 'make USE_VAGRANT=0 -C /vagrant shell'

tail:
	@${VAGRANT} ssh -c 'make USE_VAGRANT=0 -s -C /vagrant tail'

browser:
	@${VAGRANT} ssh -c 'cat ~/.local/share/jupyter/runtime/jpserver-*.json' \
	| ${PYTHON3} -c ${BROWSER_SCRIPT}

endif # USE_VAGRANT
