#!/usr/bin/env bash
# Compatibilidad: delega al target make benchmark.
exec make benchmark "$@"
