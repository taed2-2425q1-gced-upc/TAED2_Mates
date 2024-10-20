# plugin_interface.py

class PluginInterface:
    def generate_output(self, output_directory, filename):
        raise NotImplementedError("Subclasses must implement this method")
