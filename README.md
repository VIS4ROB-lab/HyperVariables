<div align="center">
  <a href="https://github.com/VIS4ROB-lab/HyperVariables">
    <img src="https://drive.google.com/uc?export=view&id=1UAFr3tepqKwdnTomhKaeI2eIag3HOISY" alt="" style="width: 150px;">
  </a>

<h2><em>Hyper</em>Variables</h2>
  <p>
    General variable and Lie group containers with extensive Eigen interoperability for non-linear optimizations.
    <br />
    <a href="https://github.com/VIS4ROB-lab/HyperVariables/issues">Report Issues or Request Features</a>
  </p>
</div>
<br />

## About

[*Hyper*Variables](https://github.com/VIS4ROB-lab/HyperVariables) comprises low-level containers used in
[*Hyper*SLAM](https://github.com/VIS4ROB-lab/HyperSLAM) to represent parameters within non-linear optimizations. All
implemented variables are fully interoperable with [Eigen](https://eigen.tuxfamily.org/) and, as such,
are mappable from raw memory locations. All variables inherit from an abstract base, which allows
storing non-homogeneous variables in a simple and unified manner. At present, three different distortion models along
with two common Lie groups and other common variables are implemented. Where adequate, we also provide analytic
Jacobians for the implemented operations. If you use this repository, please cite it as below.

```
@article{RAL2022Hug,
    author={Hug, David and B\"anninger, Philipp and Alzugaray, Ignacio and Chli, Margarita},
    journal={IEEE Robotics and Automation Letters},
    title={Continuous-Time Stereo-Inertial Odometry},
    year={2022},
    volume={7},
    number={3},
    pages={6455-6462},
    doi={10.1109/LRA.2022.3173705}
}
```

***Note:*** Development on HyperSLAM-related repositories has been discontinued.

## Installation

[*Hyper*Variables](https://github.com/VIS4ROB-lab/HyperVariables) depends on
the [Eigen](https://eigen.tuxfamily.org/), [Google Logging](https://github.com/google/glog) and
[Google Test](https://github.com/google/googletest) libraries and uses features from the
[C++20](https://en.cppreference.com/w/cpp/20) standard (see
[link](https://askubuntu.com/questions/26498/how-to-choose-the-default-gcc-and-g-version) to update gcc and g++
alternatives). The setup process itself (without additional compile flags) is as follows:

```
# Clone repository.
git clone https://github.com/VIS4ROB-lab/HyperVariables.git && cd HyperVariables/

# Run installation.
chmod +x install.sh
sudo install.sh

# Build repository.
mkdir build && cd build
cmake ..
make
```

## Literature
1. [Continuous-Time Stereo-Inertial Odometry, Hug et al. (2022)](https://ieeexplore.ieee.org/document/9772323)
2. [HyperSLAM: A Generic and Modular Approach to Sensor Fusion and Simultaneous<br /> Localization And Mapping in Continuous-Time, Hug and Chli (2020)](https://ieeexplore.ieee.org/document/9320417)
3. [A Micro Lie Theory for State Estimation in Robotics, Solà et al. (2018)](https://arxiv.org/abs/1812.01537)
4. [A Generic Camera Model and Calibration Method for Conventional,<br /> Wide-Angle, and Fish-Eye Lenses, Kannala and Brandt (2006)](https://ieeexplore.ieee.org/document/1642666)
5. [Single View Point Omnidirectional Camera Calibration from Planar Grids, Mei and Rives (2007)](https://ieeexplore.ieee.org/document/4209702)

### Known Issues

1. Tests covering the (equidistant) distortion models occasionally fail due to surpassing the numeric tolerances.

### Updates

17.06.22 Initial release of *Hyper*Variables.

### Contact

Admin - [David Hug](mailto:dhug@ethz.ch), Leonhardstrasse 21, 8092 Zürich, ETH Zürich, Switzerland  
Maintainer - [Philipp Bänninger](mailto:baephili@ethz.ch), Leonhardstrasse 21, 8092 Zürich, ETH Zürich, Switzerland  
Maintainer - [Ignacio Alzugaray](mailto:aignacio@ethz.ch), Leonhardstrasse 21, 8092 Zürich, ETH Zürich, Switzerland

### License

*Hyper*Variables are distributed under the [BSD-3-Clause License](LICENSE).
