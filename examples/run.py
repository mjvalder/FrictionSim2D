#!/usr/bin/env python3
"""Simple runner for FrictionSim2D

Usage:    
    # Run all simulations defined in the config
    afm("afm_config.ini")
"""

from frictionsim2d import afm

if __name__ == "__main__":
    afm("afm_config.ini")
