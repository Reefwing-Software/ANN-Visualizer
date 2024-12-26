"""
A visualizer application for simple deep neural networks.
"""

import toga, sys, logging, json, os
import numpy as np

from toga.style import Pack
from toga.style.pack import COLUMN, ROW, CENTER

from pathlib import Path
from logging.handlers import RotatingFileHandler
from tensorflow.keras.models import load_model

# Assign log destination (production=True, log to file and production=False, log to console)
PRODUCTION = False

def setup_logging(file_path, production=False):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    logger.handlers = []  # Remove any existing handlers
    
    if production:
        # Production mode: log to a file - Create a RotatingFileHandler
        handler = RotatingFileHandler(file_path, maxBytes=1024, backupCount=3)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG) 
    else:
        # Development mode: log to console with a simpler format (default setup)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)  # More verbose level for debugging

class Colors:
    BLACK = "#000000"
    WHITE = "#FFFFFF"
    LIGHT_BLUE = "#ADD8E6"
    DARK_BLUE = "#00008B"
    RED = "#FF0000"
    GREEN = "#00FF00"
    LIGHT_GREEN = "#90EE90"
    DARK_GREEN = "#006400"
    BLUE = "#0000FF"
    ORANGE = "#FFA500"
    MAGENTA = "#FF00FF"

class VisualDNN(toga.App):
    def startup(self):
        log_path = str(self.paths.app / 'logs/visualdnn.log')
        setup_logging(file_path=log_path, production=PRODUCTION)
        python_version_info = sys.version_info

        # DEBUG Information
        logging.info("Starting Visual ANN...")
        logging.info("====================================================================")
        logging.info("Platform: " + toga.platform.current_platform)
        logging.info("Toga v" + toga.__version__)
        logging.info("Visual ANN v" + self.version)
        logging.info(f"Python v{python_version_info.major}.{python_version_info.minor}.{python_version_info.micro}")
        logging.info("====================================================================")

        # Define input grid properties
        self.rows = 28
        self.cols = 28
        self.cell_size = 10
        self.top_left_x = 20
        self.top_left_y = 120

        # Instance variables
        self.recent_files = []
        self.file_edited = False
        self.file_loading = False

        # Create a 2D array to store cell states (0: white, 1: black)
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        
        # Configure UI elements
        self.main_window = toga.MainWindow(title="Visual ANN ", size = (1024, 450))

        # Create canvas
        self.canvas = toga.Canvas(
            style=Pack(flex=1),
            on_press=self.on_mouse_down,
            on_drag=self.on_mouse_move,
            on_release=self.on_mouse_up,
        )

        self.prediction_label = toga.Label(
            text="Prediction: N/A",  
            style=Pack(padding_top=50, text_align="center")  
        )

        # Input Box: Contains the grid and Clear button
        input_box = toga.Box(
            children=[
                self.canvas,
                toga.Button(
                    text="Clear",
                    on_press=self.clear_grid,
                    style=Pack(width=70, padding_top=10, padding_left=20, padding_bottom=100)
                ),
                self.prediction_label 
            ],
            style=Pack(direction=COLUMN, padding=10, alignment="center", width = self.cols * self.cell_size + 20)
        )

        # Neural Network Layers
        layer_1_box = self.create_neuron_layer_box(
            layer_name="Layer 1",
            neuron_color=Colors.MAGENTA,  
            layer_id=1,
            neuron_count=25,
            neuron_size=20,
            gap=1
        )

        layer_2_box = self.create_neuron_layer_box(
            layer_name="Layer 2",
            neuron_color=Colors.ORANGE,  
            layer_id=2,
            neuron_count=25,
            neuron_size=20,
            gap=1
        )

        self.clear_all_neuron_boxes()

        # Output Box
        output_box = self.create_output_box(box_size=40, gap=5)
        self.clear_output_box()

        # Network Connections Image
        net_image = toga.Image(self.paths.app/ "resources/connections.png")
        view = toga.ImageView(
            net_image,
            style=Pack(width=400, padding_top=40),
        )

        # Main Layout: Horizontal arrangement of input, layers, and output
        main_layout = toga.Box(
            children=[input_box, layer_1_box, view, layer_2_box, output_box],
            style=Pack(direction=ROW, padding=10)
        )

        self.main_window.content = main_layout

        # Commands/Menu Items for 'File' group
        self.file_sub_menu = toga.Group("Open Recent", parent=toga.Group.FILE, order=0)

        file_new_cmd = toga.Command(
            self.action_new_file, 
            text='New File', 
            shortcut=toga.Key.MOD_1 + 'n',
            group=toga.Group.FILE
        )

        file_open_cmd = toga.Command(
            self.action_open_file_dialog, 
            text='Open', 
            shortcut=toga.Key.MOD_1 + 'o',
            group=toga.Group.FILE
        )

        file_save_cmd = toga.Command(
            self.action_save_file_dialog, 
            text='Save', 
            shortcut=toga.Key.MOD_1 + 's',
            group=toga.Group.FILE
        )

        # Adding commands to the menu (not the toolbar)
        self.commands.add(file_new_cmd, file_open_cmd, file_save_cmd)

        # Draw the initial grid
        self.draw_grid(self.top_left_x, self.top_left_y)

        # Load the neural network model
        model_path = self.paths.app / 'model/mnist_model.keras'
        layer_1_model_path = self.paths.app / 'model/hidden_layer_1_model.keras'
        layer_2_model_path = self.paths.app / 'model/hidden_layer_2_model.keras'

        if model_path.exists() and layer_1_model_path.exists() and layer_2_model_path.exists():
            self.model = load_model(model_path)
            logging.info("Loaded the pre-trained model.")
            self.model.summary()

            # Load the first hidden layer model
            self.layer_1_model = load_model(layer_1_model_path)
            logging.info("Loaded the hidden layer 1 model.")

            # Load the second hidden layer model
            self.layer_2_model = load_model(layer_2_model_path)
            logging.info("Loaded the hidden layer 2 model.")
        else:
            logging.error("Model file(s) not found. Please train the model and save all required files.")
            sys.exit()

        # Main window & Exit Handler of the application
        self.on_exit = self.exit_handler
        self.main_window.show()
        self.load_recent_files()

    async def action_new_file(self, widget):
        if self.file_edited:
            open_new_file = await self.main_window.confirm_dialog("Warning - Unsaved Changes", "Are you sure you want to create a new file? Cancel to return.")
            if not open_new_file:
                return False
            
        self.clear_grid(widget)

    async def action_open_file_dialog(self, widget):
        if self.file_edited:
            open_new_file = await self.main_window.confirm_dialog("Warning - Unsaved Changes", "Are you sure you want to open a new file? Cancel to return.")
            if not open_new_file:
                return False
            
        try:
            dialog = toga.OpenFileDialog(title="Load Grid", file_types=["json"])
            file_path = await self.main_window.dialog(dialog)
            
            if file_path is not None:
                with open(file_path, "r") as file:
                    grid_1d = json.load(file)
                # Reshape back to 2D
                self.grid = np.array(grid_1d).reshape(self.rows, self.cols).tolist()
                self.draw_grid(self.top_left_x, self.top_left_y)

                # Show confirmation
                info_dialog = toga.InfoDialog(
                    title="Grid Loaded", message=f"Grid loaded from {file_path}"
                )
                await self.main_window.dialog(info_dialog)
                if file_path not in self.recent_files:
                    self.recent_files.insert(0, file_path)
                self.recent_files = self.recent_files[:5]  # Keep only the 5 most recent
                self.build_menus()  # Rebuild the recent files menu

        except ValueError as e:
            logging.warning(f"Failed to open the file due to an error: {e}")
    
    async def action_save_file_dialog(self, widget):
        file_name = "grid.json"
        try:
            dialog = toga.SaveFileDialog(title="Save Grid", suggested_filename="grid.json", file_types=["json"])
            save_path = await self.main_window.dialog(dialog)
            
            if save_path is not None:
                # Flatten the grid to 1D
                grid_1d = np.array(self.grid).flatten().tolist()
                with open(save_path, 'w') as file:
                    json.dump(grid_1d, file)
                info_dialog = toga.InfoDialog(title="Grid Saved", message=f"Grid saved to {save_path}")
                await self.main_window.dialog(info_dialog)

                self.file_loading = True
                self.file_edited = False

        except ValueError:
            logging.warning("Save file dialog was canceled")

    def load_and_render_file(self, file_path):
        self.file_loading = True
        self.file_edited = False
        if file_path is not None:
            with open(file_path, "r") as file:
                grid_1d = json.load(file)
            
            # Reshape back to 2D
            self.grid = np.array(grid_1d).reshape(self.rows, self.cols).tolist()
            self.draw_grid(self.top_left_x, self.top_left_y)

    def command_exists(self, text):
        for cmd in self.commands:
            if isinstance(cmd, toga.Command) and cmd.text == text:
                return True
        return False

    def build_menus(self):
        for file_path in self.recent_files:
            file_name = os.path.basename(file_path)  # Extract file name and extension
            if not self.command_exists(file_name):
                # Bind the current value of file_path to the lambda function
                cmd_action = lambda widget, path=file_path: self.open_recent_file(path)
                cmd = toga.Command(
                    cmd_action,
                    text=file_name,
                    group=self.file_sub_menu
                )
                self.commands.add(cmd)

    def open_recent_file(self, file_path):
        if self.file_edited:
            self.main_window.info_dialog("Warning - Unsaved Changes", "Are you sure you want to open a new file?")

        self.load_and_render_file(file_path)

    def load_recent_files(self):
        # Construct the full path for the file
        file_path = self.paths.app / 'data/recent_files.txt'
        
        # Check if the file exists and is not empty
        if file_path.exists() and file_path.stat().st_size > 0:
            # Open and read the file contents into self.recent_files
            with open(file_path, 'r') as file:
                # Convert each line from string to PosixPath and load into self.recent_files
                self.recent_files = [Path(line.strip()) for line in file.readlines()]

            self.build_menus()

    def save_recent_files(self):
        # Check if the list is not empty
        if self.recent_files:
            # Construct the full path for the file
            file_path = self.paths.app / 'data/recent_files.txt'
            
            # Write the recent files to the text file
            with open(file_path, 'w') as file:
                for file_name in self.recent_files:
                    file_name_str = str(file_name)
                    file.write(file_name_str + '\n')

    def create_neuron_layer_box(self, layer_name, neuron_color, layer_id, neuron_count=25, neuron_size=20, gap=5):
        """
        Create a box representing a neural network layer with neurons.

        Args:
            layer_name (str): The name of the layer to display as a label.
            neuron_color (str): The color of the neuron outlines.
            layer_id (int): A unique identifier for the layer.
            neuron_count (int): The number of neurons in the layer.
            neuron_size (int): The size of each neuron box.
            gap (int): The gap between neuron boxes.

        Returns:
            toga.Box: A box containing the labeled neurons for this layer.
        """
        # Ensure we have a dictionary to store neurons by layer
        if not hasattr(self, "neuron_boxes_by_layer"):
            self.neuron_boxes_by_layer = {}

        # Store references to this layer's neurons
        self.neuron_boxes_by_layer[layer_id] = []

        # Create the parent box for the layer
        layer_box = toga.Box(
            children=[
                toga.Label(layer_name, style=Pack(padding_left=15, padding_bottom=10))  
            ],
            style=Pack(direction=COLUMN, padding=10, alignment="center", flex=1)
        )

        # Add neurons as small unfilled boxes
        for _ in range(neuron_count):
            neuron_canvas = toga.Canvas(
                style=Pack(width=neuron_size, height=neuron_size, padding_bottom=gap, alignment="center")
            )

            # Draw an unfilled rectangle to represent the neuron
            with neuron_canvas.context.Stroke(color=neuron_color, line_width=2) as stroke:
                stroke.rect(x=0, y=0, width=neuron_size, height=neuron_size)

            # Store the canvas reference for this layer
            self.neuron_boxes_by_layer[layer_id].append(neuron_canvas)

            # Add the neuron canvas to the layer box
            layer_box.add(neuron_canvas)

        return layer_box

    def create_output_box(self, box_size=40, gap=10):
        """Create an output box with 11 labeled boxes."""
        # Create the parent box
        self.output_boxes = []  # Store canvas references for each box
        output_box = toga.Box(
            children=[
                toga.Label("Output", style=Pack(padding_top=15, padding_bottom=10, text_align="left"))
            ],
            style=Pack(direction=COLUMN, padding=10, alignment="center", flex=1)
        )

        # Labels for output boxes
        output_labels = [str(i) for i in range(10)] + ["᎒᎒᎒"]

        # Add boxes and labels
        for label_text in output_labels:
            # Create a horizontal box to hold the output box and its label
            row_box = toga.Box(
                style=Pack(direction=ROW, alignment="center", padding_bottom=gap)
            )

            # Tweak last output box position
            padding = 4 if label_text == "᎒᎒᎒" else 0

            # Create the output box canvas
            output_canvas = toga.Canvas(
                style=Pack(
                    width=box_size,
                    height=box_size,
                    padding_left=padding,
                    padding_right=10,  # Add some space between the box and label
                )
            )

            # Draw an empty stroked rectangle for the box
            with output_canvas.context.Stroke(color=Colors.DARK_GREEN, line_width=2) as stroke:
                stroke.rect(x=0, y=0, width=box_size, height=box_size)

            # Create the label for the box
            box_label = toga.Label(
                label_text,
                style=Pack(padding_left=10, alignment="left")
            )

            # Store the canvas reference for dynamic updates
            self.output_boxes.append(output_canvas)

            # Add the canvas (box) and label to the row
            row_box.add(output_canvas)
            row_box.add(box_label)

            # Add the row to the parent output box
            output_box.add(row_box)

        return output_box

    def clear_output_box(self):
        """
        Set the fill of all boxes to 0% except for box 10, which is set to 100% fill.
        """
        # Loop through all boxes
        for i in range(len(self.output_boxes)):
            if i == 10:  # Box 10
                self.update_output_box_fill(box_number=i, fill_percentage=1.0, fill_color=Colors.DARK_GREEN)
            else:  # All other boxes
                self.update_output_box_fill(box_number=i, fill_percentage=0.0)

    def clear_all_neuron_boxes(self):
        """
        Clear all neuron boxes by setting their fill percentage to 0 for all layers,
        with default stroke colors for each layer.
        """
        if not hasattr(self, "neuron_boxes_by_layer"):
            logging.error("No neuron layers have been created.")
            return

        # Define default stroke colors for each layer
        layer_stroke_colors = {
            1: Colors.MAGENTA,  # Stroke color for layer 1
            2: Colors.ORANGE    # Stroke color for layer 2
        }

        for layer_id, neuron_boxes in self.neuron_boxes_by_layer.items():
            # Get the default stroke color for the layer, fallback to a neutral color if not defined
            stroke_color = layer_stroke_colors.get(layer_id, Colors.DARK_GREEN)

            for neuron_number in range(len(neuron_boxes)):
                # Set the fill percentage to 0 and use the default stroke color
                self.update_neuron_fill(
                    layer_id=layer_id,
                    neuron_number=neuron_number,
                    fill_percentage=0.0,
                    fill_color=Colors.LIGHT_BLUE  # Optional: Use a neutral fill color for cleared neurons
                )

                # Update the stroke color to match the default for the layer
                neuron_canvas = self.neuron_boxes_by_layer[layer_id][neuron_number]
                with neuron_canvas.context.Stroke(color=stroke_color, line_width=2) as stroke:
                    stroke.rect(x=0, y=0, width=neuron_canvas.style.width, height=neuron_canvas.style.height)

                # Redraw the neuron canvas to apply the changes
                neuron_canvas.redraw()

    def update_output_box_fill(self, box_number, fill_percentage, fill_color=Colors.DARK_GREEN):
        """
        Update the fill of a specific output box by percentage, filling from left to right.
        Args:
            box_number (int): The index of the box to update (0-10).
            fill_percentage (float): The percentage of the box to fill (0.0 to 1.0).
            fill_color (str): The color to fill the box.
        """
        if 0 <= box_number < len(self.output_boxes):
            box_canvas = self.output_boxes[box_number]
            box_size = box_canvas.style.width

            # Calculate the fill width based on the percentage
            fill_width = box_size * fill_percentage

            # Remove existing fills if any
            box_canvas.context.clear()

            # Redraw the stroked rectangle
            with box_canvas.context.Stroke(color=Colors.DARK_GREEN, line_width=2) as stroke:
                stroke.rect(x=0, y=0, width=box_size, height=box_size)

            # Add the filled rectangle (filling left to right)
            with box_canvas.context.Fill(color=fill_color) as fill:
                fill.rect(x=0, y=0, width=fill_width, height=box_size)

            # Request a redraw
            box_canvas.redraw()

    def update_neuron_fill(self, layer_id, neuron_number, fill_percentage, fill_color=Colors.MAGENTA):
        """
        Update the fill of a specific neuron in a specific layer by percentage, filling left to right.

        Args:
            layer_id (int): The identifier of the layer containing the neuron.
            neuron_number (int): The index of the neuron to update (0-based).
            fill_percentage (float): The percentage of the neuron to fill (0.0 to 1.0).
            fill_color (str): The color to fill the neuron.
        """
        if layer_id in self.neuron_boxes_by_layer and 0 <= neuron_number < len(self.neuron_boxes_by_layer[layer_id]):
            neuron_canvas = self.neuron_boxes_by_layer[layer_id][neuron_number]
            neuron_size = neuron_canvas.style.width

            # Calculate the fill width based on the percentage
            fill_width = neuron_size * fill_percentage

            # Remove existing fills if any
            neuron_canvas.context.clear()

            # Redraw the stroked rectangle
            with neuron_canvas.context.Stroke(color=fill_color, line_width=2) as stroke:
                stroke.rect(x=0, y=0, width=neuron_size, height=neuron_size)

            # Add the filled rectangle (filling left to right)
            with neuron_canvas.context.Fill(color=fill_color) as fill:
                fill.rect(x=0, y=0, width=fill_width, height=neuron_size)

            # Request a redraw
            neuron_canvas.redraw()

    def update_output_boxes_from_predictions(self, prediction_percentage_array):
        """
        Update the fill of boxes 0-9 based on the prediction probabilities.
        Box 10 (unknown) is set to 0% fill.
        
        Args:
            prediction_percentage_array (list or np.array): Array of prediction probabilities for boxes 0-9.
        """
        # Loop through boxes 0-9 and set their fill percentage based on predictions
        for i in range(10):
            fill_percentage = prediction_percentage_array[i]  # Get the prediction probability (0.0 to 1.0)
            self.update_output_box_fill(box_number=i, fill_percentage=fill_percentage, fill_color=Colors.DARK_GREEN)
        
        # Set box 10 to 0% fill
        self.update_output_box_fill(box_number=10, fill_percentage=0.0)

    def update_neuron_boxes_from_predictions(self, layer_id, prediction_array, max_value=10.0, fill_color=Colors.MAGENTA):
        """
        Update the fill of neuron boxes in a specific layer based on the model predictions.

        Args:
            layer_id (int): The identifier of the layer (e.g., 1 for the first hidden layer, 2 for the second hidden layer).
            prediction_array (list or np.array): Array of neuron activations for the layer.
            max_value (float): The maximum activation value that corresponds to 100% fill.
            fill_color (str): The color to fill the neuron boxes.
        """
        if layer_id not in self.neuron_boxes_by_layer:
            logging.error(f"Layer ID {layer_id} does not exist.")
            return

        neuron_boxes = self.neuron_boxes_by_layer[layer_id]
        neuron_count = len(neuron_boxes)

        if len(prediction_array) != neuron_count:
            logging.error(f"Mismatch between neuron count ({neuron_count}) and prediction array length ({len(prediction_array)}).")
            return

        # Update each neuron box based on its activation value
        for neuron_number, activation_value in enumerate(prediction_array):
            # Normalize the activation value to a percentage (0.0 to 1.0)
            fill_percentage = min(max(activation_value / max_value, 0.0), 1.0)

            # Update the specific neuron fill
            self.update_neuron_fill(
                layer_id=layer_id,
                neuron_number=neuron_number,
                fill_percentage=fill_percentage,
                fill_color=fill_color
            )

    def clear_grid(self, widget):
        """Clear all filled cells in the input grid."""
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
        self.draw_grid(self.top_left_x, self.top_left_y)
        self.clear_output_box()
        self.clear_all_neuron_boxes()
        self.prediction_label.text = "Prediction: N/A"

    def draw_grid(self, top_left_x=0, top_left_y=0):
        """Draw the grid on the canvas starting at (top_left_x, top_left_y)."""
        self.canvas.context.clear()

        # Draw each cell in the input grid
        for row in range(self.rows):
            for col in range(self.cols):
                # Calculate the position of the current cell
                x1 = top_left_x + col * self.cell_size
                y1 = top_left_y + row * self.cell_size

                # Map grayscale value (0–255) to a hex color
                intensity = int(self.grid[row][col])  # Get intensity (0–255)
                cell_color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"  # Grayscale hex color

                # Draw the cell
                with self.canvas.context.Fill(color=cell_color) as fill:
                    fill.rect(x=x1, y=y1, width=self.cell_size, height=self.cell_size)

                # Draw the cell border in white
                with self.canvas.context.Stroke(color=Colors.WHITE, line_width=1) as stroke:
                    stroke.rect(x=x1, y=y1, width=self.cell_size, height=self.cell_size)
        
    def update_predictions(self):
        # Flatten the 28x28 grid into a 1D array and normalize values to [0, 1]
        input_data = np.array(self.grid).astype('float32').reshape(1, 784) / 255.0

        # Make a prediction using the loaded model
        prediction = self.model.predict(input_data)
        prediction_percentage_array = prediction[0]
        self.update_output_boxes_from_predictions(prediction_percentage_array)

        return prediction, prediction_percentage_array

    def intermediate_predictions(self):
        input_data = np.array(self.grid).astype('float32').reshape(1, 784) / 255.0

        layer_1_predictions = self.layer_1_model.predict(input_data)[0]  
        layer_2_predictions = self.layer_2_model.predict(input_data)[0]  

        # Update neuron boxes for both hidden layers
        self.update_neuron_boxes_from_predictions(
            layer_id=1,
            prediction_array=layer_1_predictions,
            max_value=10.0,
            fill_color=Colors.MAGENTA
        )

        self.update_neuron_boxes_from_predictions(
            layer_id=2,
            prediction_array=layer_2_predictions,
            max_value=10.0,
            fill_color=Colors.ORANGE
        )

    def on_mouse_down(self, widget, x, y):
        """Handle mouse down events to fill one cell."""
        self.fill_cell_at(x, y)
        self.intermediate_predictions()
        self.update_predictions()

    def on_mouse_move(self, widget, x, y):
        """Handle mouse drag events to fill multiple cells."""
        self.fill_cell_at(x, y)
        self.intermediate_predictions()
        self.update_predictions()

    def on_mouse_up(self, widget, x, y):
        self.intermediate_predictions()
        prediction, prediction_percentage_array = self.update_predictions()

        # Get the class with the highest probability
        predicted_class = np.argmax(prediction)
        predicted_percentage = prediction_percentage_array[predicted_class] * 100  # Convert to percentage
        print(f"Predicted class: {predicted_class} with {predicted_percentage:.2f}% confidence")

        self.prediction_label.text = f"Prediction: {predicted_class} with {predicted_percentage:.2f}%  confidence"

    def fill_cell_at(self, x, y):
        """Fill the cell under the mouse position with antialiasing."""
        col = int((x - self.top_left_x) // self.cell_size)
        row = int((y - self.top_left_y) // self.cell_size)

        # Check if the mouse is within the grid bounds
        if 0 <= row < self.rows and 0 <= col < self.cols:
            # Set the main cell to white (255)
            self.grid[row][col] = 255

            # Add antialiasing effect to neighboring cells
            neighbors = [
                (row - 1, col), (row + 1, col),  # Top and bottom neighbors
                (row, col - 1), (row, col + 1),  # Left and right neighbors
            ]
            for r, c in neighbors:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    self.grid[r][c] = max(self.grid[r][c], 128)  # Set to a medium gray (128)

            # Add diagonal neighbors for smoother edges
            diagonal_neighbors = [
                (row - 1, col - 1), (row - 1, col + 1),  # Top-left, top-right
                (row + 1, col - 1), (row + 1, col + 1),  # Bottom-left, bottom-right
            ]
            for r, c in diagonal_neighbors:
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    self.grid[r][c] = max(self.grid[r][c], 64)  # Set to a light gray (64)

            # Redraw the grid
            self.draw_grid(self.top_left_x, self.top_left_y)

    async def exit_handler(self, app, **kwargs):
        if self.file_edited:
            exit_application = await self.main_window.confirm_dialog("Warning - Unsaved Changes", "Are you sure you want to exit the app? Cancel to return.")
            if not exit_application:
                return False
        
        self.save_recent_files()
        logging.debug("Application Exiting...")

        return True

    @property
    def height(self):
        return self.canvas.layout.content_height

    @property
    def width(self):
        return self.canvas.layout.content_width

def main():
    return VisualDNN()
