"""
ConanFile.
"""

from conans import ConanFile, CMake


class ConanConfig(ConanFile):
    """
    ConanConfig
    """

    requires = "doctest/2.3.1@bincrafters/stable"

    generators = "cmake"

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()
