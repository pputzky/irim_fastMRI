import sys, os
dirname = os.path.dirname(__file__)

external_path = os.path.abspath(os.path.join(dirname, '..','external'))
sys.path.insert(0, external_path)