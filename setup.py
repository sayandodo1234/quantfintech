#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 18:40:23 2021

@author: sayandebhowmick
"""


from distutils.core import setup
setup(
  name = 'quantfintech',         
  packages = ['quantfintech'],   
  version = '0.1',      
  license='MIT',       
  description = 'A libraray for quantitative finance and algo trading.',   
  author = 'Sayan De Bhowmick',                   
  author_email = 'debhowmick.sayan@gmail.com',      
  url = 'https://github.com/sayandodo1234/quantfintech',   
  download_url = 'https://github.com/sayandodo1234/quantfintech/archive/refs/tags/v_01.3.zip',    
  keywords = ['Technical Indicators', 'Quant Analysis', 'Algo trading'],
  install_requires=[           
          'pandas',
          'numpy',
          'statsmodels'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
