#!/usr/bin/env bash
set -Eeuo pipefail

# check to see if this file is being run or sourced from another script
_is_sourced() {
	# https://unix.stackexchange.com/a/215279
	[ "${#FUNCNAME[@]}" -ge 2 ] &&
		[ "${FUNCNAME[0]}" = '_is_sourced' ] &&
		[ "${FUNCNAME[1]}" = 'source' ]
}

_main() {
	# For backwards compatibility with the yatai<1.0.0, adapting the old "yatai" command to the new "start" command.
	if [ "${#}" -gt 0 ] && [ "${1}" = 'python' ] && [ "${2}" = '-m' ] && { [ "${3}" = 'vtsserving._internal.server.cli.runner' ] || [ "${3}" = "vtsserving._internal.server.cli.api_server" ]; }; then # SC2235, use { } to avoid subshell overhead
		if [ "${3}" = 'vtsserving._internal.server.cli.runner' ]; then
			set -- vtsserving start-runner-server "${@:4}"
		elif [ "${3}" = 'vtsserving._internal.server.cli.api_server' ]; then
			set -- vtsserving start-http-server "${@:4}"
		fi
	# If no arg or first arg looks like a flag.
	elif [[ "$#" -eq 0 ]] || [[ "${1:0:1}" =~ '-' ]]; then
		# This is provided for backwards compatibility with places where user may have
		# discover this easter egg and use it in their scripts to run the container.
		if [[ -v VTSSERVING_SERVE_COMPONENT ]]; then
			echo "\$VTSSERVING_SERVE_COMPONENT is set! Calling 'vtsserving start-*' instead"
			if [ "${VTSSERVING_SERVE_COMPONENT}" = 'http_server' ]; then
				set -- vtsserving start-http-server "$@" "$VTS_PATH"
			elif [ "${VTSSERVING_SERVE_COMPONENT}" = 'grpc_server' ]; then
				set -- vtsserving start-grpc-server "$@" "$VTS_PATH"
			elif [ "${VTSSERVING_SERVE_COMPONENT}" = 'runner' ]; then
				set -- vtsserving start-runner-server "$@" "$VTS_PATH"
			fi
		else
			set -- vtsserving serve --production "$@" "$VTS_PATH"
		fi
	fi
	# Overide the VTSSERVING_PORT if PORT env var is present. Used for Heroku and Yatai.
	if [[ -v PORT ]]; then
		echo "\$PORT is set! Overiding \$VTSSERVING_PORT with \$PORT ($PORT)"
		export VTSSERVING_PORT=$PORT
	fi
	# Handle serve and start commands that is passed to the container.
	# Assuming that serve and start commands are the first arguments
	# Note that this is the recommended way going forward to run all vtsserving containers.
	if [ "${#}" -gt 0 ] && { [ "${1}" = 'serve' ] || [ "${1}" = 'serve-http' ] || [ "${1}" = 'serve-grpc' ] || [ "${1}" = 'start-http-server' ] || [ "${1}" = 'start-grpc-server' ] || [ "${1}" = 'start-runner-server' ]; }; then
		exec vtsserving "$@" "$VTS_PATH"
	else
		# otherwise default to run whatever the command is
		# This should allow running bash, sh, python, etc
		exec "$@"
	fi
}

if ! _is_sourced; then
	_main "$@"
fi
