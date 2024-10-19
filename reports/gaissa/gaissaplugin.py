import os
from plugin_interface import PluginInterface
from calculator import calculateH5

class PlugIn(PluginInterface):
    def generate_output(self, model_path, output_directory, filename):        
        # Create the full path for the output file
        output_path = os.path.join(output_directory, filename + '.csv')
        
        calculateH5(model_path, output_path)

        print(f"Output file '{filename}' generated in '{output_directory}'")
