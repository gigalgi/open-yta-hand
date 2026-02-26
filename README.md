# open-yta-hand

**A fully underactuated, monolithic anthropomorphic hand with integrated wrist — 7 DOF per finger, cable-driven, 3D printed, impedance controlled.**

[![Thesis](https://img.shields.io/badge/Thesis-SKKU%202024-blue)](https://dcollection.skku.edu/srch/srchDetail/000000181091?localeParam=en)
[![ICCAS 2024](https://img.shields.io/badge/Paper-ICCAS%202024-green)](https://ieeexplore.ieee.org/document/10773068)
[![Framework](https://img.shields.io/badge/Design%20Foundation-monolithic--robotics-orange)](https://github.com/gigalgi/monolithic-robotics)
![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![License: CC BY 4.0](https://img.shields.io/badge/CAD%20Files-CC%20BY%204.0-lightgrey.svg)

## License

- Source code: Apache License 2.0 (see LICENSE)
- CAD files and design assets: Creative Commons Attribution 4.0 International (CC BY 4.0)

---

> *Yta* — "hand" in Muisca, the language of the pre-Columbian Chibcha civilization of the Colombian highlands.

---

## What This Is

Open-Yta-Hand is the full implementation of the [Monolithic Robotics](../monolithic-robotics) hand framework: every mechanical component, all control software, and the complete sensing stack — from a single 3D-printed finger to an assembled hand with wrist.

The hand features four underactuated fingers and an opposable thumb, each built from the UMoBIC joint — a FACT-derived compliant revolute unit described in detail in the [monolithic-robotics](../monolithic-robotics) repository. This repository contains everything needed to build, assemble, and run a working prototype: CAD files, firmware, kinematics, simulation, and control code.
