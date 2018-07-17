# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 10:32:21 2018

@author: Tim Rodgers
"""
import math

class Pizza(object):
    def __init__(self, radius, height):
        self.radius = radius
        self.height = height

    @staticmethod
    def compute_area(radius):
         return math.pi * (radius ** 2)

    @classmethod
    def compute_volume(cls, height, radius):
         return height * cls.compute_area(radius)

    def get_volume(self):
        return self.compute_volume(self.height, self.radius)
    