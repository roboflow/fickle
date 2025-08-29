# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.2] - 2023-04-15

### Added

- `ext.pytorch`: Add attribute `StoredTensor.has_trivial_layout` to check whether a tensor has nontrivial stride.
- `ext.pytorch`: Factor out method `StoredTensor.open_storage()`.

## [0.2.1] - 2023-04-11

### Added

- Expose `firewall.Unknown*` in the top level package.
- `ext.pytorch`: Support more PyTorch/numpy data types.
- `ext.pytorch`: Add `StoredTensor.buffer` property to access the raw bytes of the tensor.
