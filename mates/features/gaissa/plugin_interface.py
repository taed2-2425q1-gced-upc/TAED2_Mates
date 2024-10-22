"""
Module for defining the PluginInterface.

This module provides an abstract base class, `PluginInterface`, which 
serves as a template for creating plugins that generate output based on 
a model. The class enforces a standard method, `generate_output`, that 
subclasses must implement to define their specific output generation 
logic.

Dependencies:
-------------
None.

Classes:
--------
- PluginInterface: Abstract base class for plugins that generate output.

Usage:
------
Subclasses should inherit from `PluginInterface` and implement the 
`generate_output` method to create a specific output generation 
functionality based on the model provided.
"""


class PluginInterface:
    """
    Abstract base class for plugins.

    This class defines a common interface for plugins that generate output
    based on a model. Subclasses must implement the `generate_output` method.

    Methods:
    --------
    generate_output(output_directory, filename):
        Generates an output file with the specified name in the given directory.
    """

    # pylint: disable=R0903
    def generate_output(self, output_directory, filename):
        """
        Generate an output file.

        This method must be implemented by subclasses to define how the output
        is generated.

        Parameters:
        -----------
        output_directory : str
            The directory where the output file will be saved.
        filename : str
            The name of the output file to be generated (without extension).

        Raises:
        -------
        NotImplementedError:
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")
