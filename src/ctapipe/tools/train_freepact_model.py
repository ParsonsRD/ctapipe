"""
Tool for training the FreePACT model
"""
import astropy.units as u
import numpy as np
from keras import layers
from keras.models import Sequential
from tensorflow.keras.regularizers import l2

from ctapipe.core.tool import Tool
from ctapipe.core.traits import Int, IntTelescopeParameter, Path
from ctapipe.io import TableLoader
from ctapipe.reco import CrossValidator, ParticleClassifier

__all__ = [
    "TrainFreePACTModel",
]


class TrainFreePACTModel(Tool):
    """
    Tool to train a FreePACT model on dl1b/dl2 data.

    The tool first performs a cross validation to give an initial estimate
    on the quality of the estimation and then finally trains one model
    per telescope type on the full dataset.
    """

    name = "ctapipe-train-freepact-model"
    description = __doc__

    examples = """
    ctapipe-train-freepact-model \\
        -c train_freepact_model.yaml \\
        --signal gamma.dl2.h5 \\
        --background proton.dl2.h5 \\
        -o freepact_model.pkl
    """

    input_url_signal = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help="Input dl1b/dl2 file for the signal class.",
    ).tag(config=True)

    output_path = Path(
        default_value=None,
        allow_none=False,
        directory_ok=False,
        help=(
            "Output file for the trained reconstructor."
            " At the moment, pickle is the only supported format."
        ),
    ).tag(config=True)

    n_events = IntTelescopeParameter(
        default_value=None,
        allow_none=True,
        help=(
            "Total number of events to be used for training."
            " If not given, all available events will be used"
            " (considering ``signal_fraction``)."
        ),
    ).tag(config=True)

    chunk_size = Int(
        default_value=100000,
        allow_none=True,
        help=(
            "How many subarray events to load at once before training on"
            " n_events (or all available) events."
        ),
    ).tag(config=True)

    dilate_rows = Int(
        default_value=0, help="Add dilation to the rows of the camera images."
    ).tag(config=True)

    aliases = {
        "signal": "TrainFreePACTModel.input_url_signal",
        "n-events": "TrainFreePACTModel.n_events",
        "dilate-rows": "TrainFreePACTModel.dilate_rows",
        ("o", "output"): "TrainFreePACTModel.output_path",
    }

    classes = [
        TableLoader,
        ParticleClassifier,
        CrossValidator,
    ]

    def setup(self):
        """
        Initialize components from config.
        """
        self.signal_loader = self.enter_context(
            TableLoader(
                parent=self,
                input_url=self.input_url_signal,
            )
        )

        self.n_events.attach_subarray(self.signal_loader.subarray)

        self.model = self.create_model(6, number_of_layers=8, l2_val=1e-8)

        self.check_output(self.output_path)

    def start(self):
        """
        Train models per telescope type.
        """
        # By construction both loaders have the same types defined
        types = self.signal_loader.subarray.telescope_types

        self.log.info("Signal input-file: %s", self.signal_loader.input_url)
        self.log.info("Training models for %d types", len(types))

        for tel_type in types:
            self.log.info("Loading events for %s", tel_type)
            table = self._read_input_data(tel_type)

            # self.freepact.fit(tel_type, table)
            self.log.info("done")

    def _read_input_data(self, tel_type):
        return table

    def finish(self):
        """
        Write-out trained models and cross-validation results.
        """
        self.log.info("Writing output")
        self.freepact.n_jobs = None
        self.freepact.write(self.output_path, overwrite=self.overwrite)
        self.signal_loader.close()
        self.background_loader.close()
        self.cross_validate.close()

    def read_freepact_training_data(
        filename: Path,
        telescope_type: str,
        chunk_size=10000,
        dilation=4,
        nevents=None,
        offset_minimum=0.0 * u.deg,
        offset_maximum=5.0 * u.deg,
    ):
        """
        Reads and processes data from a given file for FreePACT training.

        This function loads telescope events, extracts relevant features, and returns
        them as a stacked table.

        Parameters
        ----------
        filename : pathlib.Path
            Path to the input data file.
        telescope_type : str
            The type of telescope to select events from (e.g., 'LST_LST_LSTCam').
        chunk_size : int
            The number of events to read per chunk.

        Returns
        -------
        astropy.table.Table
            A table containing the processed data with columns:
            'true_energy', 'true_impact_distance', 'true_x_max', 'image', and 'offset'.
        """
        loader = TableLoader(filename, dl1_images=True, instrument=True)
        tables = []

        # Find and read the correct camera geometry table for the requested telescope type.
        # Some files store multiple camera geometries under paths named like
        # 'configuration/instrument/telescope/camera/geometry_0', '...geometry_1', etc.
        # We try those paths in a small range until we find one whose CAM_ID matches
        # the desired telescope_type. If the path doesn't exist, QTable.read raises
        # OSError and we continue to the next index.
        for i in range(10):
            try:
                camera_table = QTable.read(
                    filename,
                    path="configuration/instrument/telescope/camera/geometry_" + str(i),
                )
            except OSError:
                # Path not present in this file, try the next index
                continue

            # camera_table.meta["CAM_ID"] may be stored as bytes, decode to string
            # and check if the camera ID (string) is part of the requested telescope_type.
            if camera_table.meta["CAM_ID"].decode("utf-8") in telescope_type:
                # Found the matching camera geometry for the telescope type; stop searching.
                break

        # Ensure metadata entries are plain Python strings (decode bytes if needed)
        for v in camera_table.meta:
            try:
                camera_table.meta[v] = camera_table.meta[v].decode("utf-8")
            except AttributeError:
                camera_table.meta[v] = camera_table.meta[v]

        # Build a CameraGeometry from the camera table and attach appropriate units.
        camera_geom = CameraGeometry.from_table(camera_table)
        camera_geom.pix_x = camera_geom.pix_x * u.m
        camera_geom.pix_y = camera_geom.pix_y * u.m
        camera_geom.pixel_width = camera_geom.pixel_width * u.m
        camera_geom.pix_area = camera_geom.pix_area * u.m**2

        # Prepare containers for the flattened training data attributes.
        pix_x, pix_y, cleaned_images, energy, impact, xmax, event_x, event_y, phi_l = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for event in loader.read_telescope_events_chunked(
            chunk_size, telescopes=[telescope_type], stop=nevents
        ):
            # Extract arrays from the event (true and reconstructed quantities)
            data = event.data
            alt = data["true_alt"]
            az = data["true_az"]
            point_alt = data["telescope_pointing_altitude"]
            point_az = data["telescope_pointing_azimuth"]

            corex = data["true_core_x"]
            corey = data["true_core_y"]

            telscope_x = data["pos_x"]
            telscope_y = data["pos_y"]
            telscope_z = data["pos_z"]

            images = data["image"]
            image_mask = data["image_mask"]

            # Build SkyCoord objects for event and pointing in the AltAz frame
            event_altaz = SkyCoord(alt=alt, az=az, frame=AltAz)
            pointing = SkyCoord(alt=point_alt, az=point_az, frame=AltAz)

            # Initialize camera frame if not already set (uses effective focal length
            # and the first telescope pointing of this chunk)
            if camera_geom.frame is None:
                camera_geom.frame = CameraFrame(
                    focal_length=data["effective_focal_length"][0] * u.m,
                    telescope_pointing=pointing[0],
                )
                camera_geom = camera_geom.transform_to(NominalFrame(origin=pointing[0]))

            # Compute offset between pointing and true event direction
            offset = pointing.separation(event_altaz)
            nominal_frame = NominalFrame(origin=pointing)
            event_nominal = event_altaz.transform_to(nominal_frame)
            tilt = TiltedGroundFrame(pointing_direction=pointing)

            # Ground-frame coordinates for core and telescope positions (used for impact distance)
            grd_core = SkyCoord(x=corex, y=corey, z=0, frame=GroundFrame, unit="m")
            grd_telescope = SkyCoord(
                x=telscope_x, y=telscope_y, z=telscope_z, frame=GroundFrame, unit="m"
            )
            tilt_core = grd_core.transform_to(tilt)
            tilt_telescope = grd_telescope.transform_to(tilt)

            # Compute azimuthal angle phi of the telescope relative to core in the tilted frame
            phi = np.arctan2(
                tilt_telescope.y - tilt_core.y, tilt_telescope.x - tilt_core.x
            )
            # 3D separation between core and telescope in the tilted frame gives impact distance
            impact_distance = grd_core.transform_to(tilt).separation_3d(
                grd_telescope.transform_to(tilt)
            )

            # Loop over telescopes/images in this event chunk
            for i in range(len(image_mask)):
                # Ensure pixel coordinates are available in degrees for masking operations
                camera_geom.pix_x.to(u.deg).value
                # Apply morphological dilation to the image mask 'dilation' times to
                # include neighboring pixels (helps include image tails)
                for j in range(dilation):
                    mask = dilate(camera_geom, image_mask[i])

                # Collect flattened pixel coordinates, pixel values, and per-pixel labels
                pix_x.extend(camera_geom.pix_x.to(u.deg).value[mask])
                pix_y.extend(camera_geom.pix_x.to(u.deg).value[mask])
                cleaned_images.extend(images[i][mask])
                # Repeat scalar event-level quantities for each selected pixel
                energy.extend(np.repeat(data["true_energy"][i], np.sum(mask)))
                xmax.extend(np.repeat(data["true_x_max"][i], np.sum(mask)))
                impact.extend(np.repeat(impact_distance[i].value, np.sum(mask)))
                event_x.extend(
                    np.repeat(event_nominal.fov_lon[i].to(u.deg).value, np.sum(mask))
                )
                event_y.extend(
                    np.repeat(event_nominal.fov_lat[i].to(u.deg).value, np.sum(mask))
                )
                phi_l.extend(np.repeat(phi[i].value, np.sum(mask)))

        # Convert lists to numpy arrays with appropriate dtypes and shapes
        pix_x = np.array(pix_x, dtype="float32").ravel()
        pix_y = np.array(pix_y, dtype="float32").ravel()
        event_x = np.array(event_x, dtype="float32").ravel()
        event_y = np.array(event_y, dtype="float32").ravel()
        phi_l = np.array(phi_l, dtype="float32")

        cleaned_images = np.array(cleaned_images).ravel()
        # Rotate and translate pixel coordinates so they are referenced relative to
        # the event location (this yields pixel positions in an event-centric frame)
        pix_x, pix_y = rotate_translate(
            pix_x.reshape(1, pix_x.shape[0]),
            pix_y.reshape(1, pix_x.shape[0]),
            event_x.reshape(1, pix_x.shape[0]),
            event_y.reshape(1, pix_x.shape[0]),
            phi_l.reshape(1, pix_x.shape[0]),
        )

        # Recompute offset from the per-pixel event coordinates and select events
        # within the requested offset range (in degrees)
        offset = np.sqrt(event_x**2 + event_y**2) * u.deg
        selection = (offset >= offset_minimum) & (offset <= offset_maximum)

        # Stack the desired columns into a single array and apply the offset selection.
        # Returned array columns: pix_x, pix_y, energy, impact, xmax, cleaned_images
        return np.array(
            (pix_x.ravel(), pix_y.ravel(), energy, impact, xmax, cleaned_images),
            dtype="float32",
        ).T[selection]

    def create_model_cnn(
        self, cnn_input_shape, filters=50, number_of_layers=14, l2_val=1e-5
    ):
        """_summary_

        Args:
            cnn_input_shape (_type_): _description_
            filters (int, optional): _description_. Defaults to 50.
            number_of_layers (int, optional): _description_. Defaults to 14.

        Returns:
            _type_: _description_
        """

        # OK first we have out CNN input layer, it's time distributed to account for the multiple telescope types

        l2_reg = l2(l2_val)

        # Fit image pixels using a multi layer perceptron
        model = Sequential()
        model.add(
            layers.Dense(
                units=filters,
                activation="swish",
                kernel_regularizer=l2_reg,
                input_shape=cnn_input_shape,
            )
        )
        # We make a very deep network
        for n in range(number_of_layers):
            model.add(
                layers.Dense(
                    units=filters, activation="swish", kernel_regularizer=l2_reg
                )
            )

        model.add(layers.Dense(units=1, activation="sigmoid"))

        return model


def main():
    TrainFreePACTModel().run()


if __name__ == "__main__":
    main()
