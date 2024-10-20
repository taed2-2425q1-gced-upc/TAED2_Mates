from gaissaplugin import PlugIn

def main():
    # Get user input for model path and output directory and filename
    model_path = input("Enter the model's file path: ")
    output_directory = input("Enter the output directory: ")
    filename = input("Enter the filename: ")

    # Instantiate the plugin
    plugin = PlugIn()

    # Call the plugin to generate the output
    plugin.generate_output(model_path, output_directory, filename)

if __name__ == "__main__":
    main()
