#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

from js import Viewer3Djs
from js import three
from js import Shapes
from js import configs
from js import gaussians
from js import RobotBuilder
from js import readFile
from js import writeFile
from pyodide.ffi import create_proxy
from pyodide.ffi import to_js
import numpy as np
import math
import time


class Viewer3D:
    """Python entry point for the 'viewer3d.js' library, used to display and interact with a 3D
    representation of the Panda robotic arm.

    This class mainly acts as a bridge between the Python and JavaScript worlds, handling all the
    needed type conversions.
    """

    def __init__(self, domElement=None, parameters=None, composition=None):
        """Constructs the 3D viewer

        Parameters:
            domElement (element): The DOM element to use for displaying the robotic arm
            parameters (dict): Additional optional parameters, to customize the behavior of the
                               3D viewer (see below)
            composition (list): Optional settings describing how to combine several rendering
                                layers (see below)
            onready (function): Function to call once all assets are loaded and the scene is ready

        If no DOM element is provided, one is created. It is the duty of the caller to insert it
        somewhere in the DOM (see 'Viewer3D.domElement').

        Optional parameters:
            joint_position_colors (list):
                the colors to use for the visual indicators around the joints (see
                "show_joint_positions", default: all 0xff0000)

            joint_position_layer (int):
                the layer on which the joint position helpers are rendered (default: 0)

            shadows (bool):
                enable the rendering of the shadows (default: true)

            show_joint_positions (bool):
                enable the display of an visual indicator around each joint (default: false)

            statistics (bool):
                enable the display of statistics about the rendering performance (default: false)

            external_loop (bool):
                indicates that the rendering frequency is controlled by the user application (default: false)

        Composition:
            3D objects can be put on different layers, each rendered on top of the previous one.
            Each layer has its own set of settings affecting the way it is rendered. Those
            settings are:

                clear_depth (bool):
                    whether to clear the depth buffer before rendering the layer (default: false)

                effect (str):
                    the effect applied to this layer. Supported values: 'outline' (default: null)

                effect_parameters (dict):
                    the parameters of the effect applied to this layer

            Example (apply the 'outline' effect on layer 1):

                [
                    {
                        'layer': 1,
                        'effect': 'outline',
                    }
                ]

            Parameters for the 'outline' effect:
                thickness (float):
                    thickness of the outline (default: 0.003)

                color (list of 4 floats):
                    RGBA color of the outline (default: [0, 0, 0, 0])
        """
        self.viewer = Viewer3Djs.new(
            domElement,
            to_js(parameters),
            to_js(composition)
        )


    @property
    def domElement(self):
        """Returns the DOM element containing the 3D viewer"""
        return self.viewer.domElement


    def loadScene(self, filename, robotBuilders=None):
        if robotBuilders is not None:
            for builder in robotBuilders:
                builder.q = to_js(builder.q)
                builder.q_offset = to_js(builder.q_offset)
                builder.alpha = to_js(builder.alpha)
                builder.d = to_js(builder.d)
                builder.r = to_js(builder.r)
                builder.defaultPose = to_js(builder.defaultPose)
                builder.colors = to_js(builder.colors)
                builder.position = to_js(builder.position)
                builder.quaternion = to_js(builder.quaternion)

        self.viewer.loadScene(filename, to_js(robotBuilders))


    @property
    def physicsSimulatorPaused(self):
        return self.viewer.physicsSimulator.paused;


    @physicsSimulatorPaused.setter
    def physicsSimulatorPaused(self, paused):
        self.viewer.physicsSimulator.paused = paused;


    def createRobot(self, name, configuration, prefix=None):
        robotjs = self.viewer.createRobot(name, to_js(configuration), prefix)
        if robotjs is None:
            return None

        return Robot(robotjs, self.viewer)


    def getRobot(self, name):
        """Returns the robot

        Returns:
            The robot (Robot)
        """
        robotjs = self.viewer.getRobot(name)
        if robotjs is None:
            return None

        return Robot(robotjs, self.viewer)


    def setRenderingCallback(self, callback):
        """Register a function that should be called once per frame.

        This callback function can for example be used to update the positions of the joints.

        The signature of the callback function is: callback(delta), with 'delta' the time elapsed
        since the last frame, in seconds.

        Note that only one function can be registered at a time. If 'callback' is 'None', no
        function is called anymore.
        """
        self.viewer.setRenderingCallback(create_proxy(callback) if callback is not None else None)


    @property
    def controlsEnabled(self):
        """Indicates if the manipulation controls are enabled

        Manipulation controls include the end-effector and the target manipulators.
        """
        self.viewer.areControlsEnabled()


    @controlsEnabled.setter
    def controlsEnabled(self, enabled):
        """Enables/disables the manipulation controls

        Manipulation controls include the end-effector and the target manipulators.
        """
        self.viewer.enableControls(enabled)


    @property
    def endEffectorManipulationEnabled(self):
        """Indicates if the manipulation of the end effector is enabled (when the user clicks
        on it).

        Note that if 'Viewer3D.controlsEnabled' is 'False', the end-effector can't be
        manipulated regardless of the value of this property.
        """
        return self.viewer.isEndEffectorManipulationEnabled()


    @endEffectorManipulationEnabled.setter
    def endEffectorManipulationEnabled(self, enabled):
        """Enables/disables the manipulation of the end effector (when the user clicks on it)

        Important: the actual Inverse Kinematics computation is expected to be done by the
        caller (in the rendering callback for instance).

        Note that if 'Viewer3D.controlsEnabled' is 'False', the end-effector can't be
        manipulated regardless of the value of this property.
        """
        self.viewer.enableEndEffectorManipulation(enabled)


    @property
    def jointsManipulationEnabled(self):
        """Indicates if the manipulation of the joint positions is enabled (when the user
        clicks on them or use the mouse wheel).

        Note that if 'Viewer3D.controlsEnabled' is 'False', the position of the joints
        can't be changed using the mouse regardless of the value of this property.
        """
        return self.viewer.isJointsManipulationEnabled()


    @jointsManipulationEnabled.setter
    def jointsManipulationEnabled(self, enabled):
        """Enables/disables the manipulation of the joint positions (when the user clicks on
        them or use the mouse wheel).

        Note that if 'Viewer3D.controlsEnabled' is 'False', the position of the joints
        can't be changed using the mouse regardless of the value of this property.
        """
        self.viewer.enableJointsManipulation(enabled)


    @property
    def linksManipulationEnabled(self):
        """Indicates if the manipulation of the links is enabled (by click and drag).

        Note that if either 'Viewer3D.controlsEnabled' or 'Viewer3D.jointsManipulationEnabled'
        are 'false', the links can't be manipulated using the mouse regardless of the
        value of this property.
        """
        return self.viewer.isLinksManipulationEnabled()


    @linksManipulationEnabled.setter
    def linksManipulationEnabled(self, enabled):
        """Enables/disables the manipulation of the links (by click and drag).

        Note that if either 'Viewer3D.controlsEnabled' or 'Viewer3D.jointsManipulationEnabled'
        are 'false', the links can't be manipulated using the mouse regardless of the
        value of this property.
        """
        self.viewer.enableLinksManipulation(enabled)


    @property
    def objectsManipulationEnabled(self):
        """Indicates if the manipulation of objects (like the gaussians) is enabled.

        Note that if 'Viewer3D.controlsEnabled' is 'False', the transforms of the
        objects can't be changed using the mouse regardless of the value of this property.
        """
        return self.viewer.isObjectsManipulationEnabled()


    @objectsManipulationEnabled.setter
    def objectsManipulationEnabled(self, enabled):
        """Enables/disables the manipulation of the objects (like the gaussians).

        Note that if 'Viewer3D.controlsEnabled' is 'False', the transforms of the
        objects can't be changed using the mouse regardless of the value of this property.
        """
        self.viewer.enableObjectsManipulation(enabled)


    @property
    def robotToolsEnabled(self):
        return self.viewer.areRobotToolsEnabled()


    @robotToolsEnabled.setter
    def robotToolsEnabled(self, enabled):
        self.viewer.enableRobotTools(enabled)


    def activateLayer(self, layer):
        """Change the layer on which new objects are created.

        Each layer is drawn on top of the previous one, after clearing the depth buffer.
        The default layer (the one were the robot is) is layer 0.

        Parameters:
            layer (int): Index of the layer
        """
        self.viewer.activateLayer(layer)


    def addTarget(self, name, position, orientation, color=None, shape=Shapes.Cube, listener=None, parameters=None):
        """Add a target to the scene, an object that can be manipulated by the user that can
        be used to define a destination position and orientation for the end-effector of the
        robot.

        Parameters:
            name (str): Name of the target
            position (list/NumPy array): The position (x, y, z) of the target
            orientation (list/NumPy array): The orientation (x, y, z, w) of the target
            color (int/str): Color of the target (by default: 0x0000aa)
            shape (Shapes): Shape of the target (by default: Shapes.Cube)
            listener (function): Function to call when the target is moved/rotated using the mouse
            parameters (dict): Additional shape-dependent parameters (radius, width, height, ...)
        """
        def _listener(targetjs, dragging):
            listener(Target(targetjs), dragging)

        if isinstance(position, np.ndarray):
            position = list(position)

        if isinstance(orientation, np.ndarray):
            orientation = list(orientation)

        return Target(
            self.viewer.addTarget(
                name,
                three.Vector3.new(*position),
                three.Quaternion.new(*orientation),
                color,
                shape,
                create_proxy(_listener) if listener is not None else None,
                to_js(parameters) if parameters is not None else None,
            )
        )


    def removeTarget(self, name):
        """Remove a target from the scene.

        Parameters:
            name (str): Name of the target
        """
        self.viewer.removeTarget(name)


    def getTarget(self, name):
        """Returns a target from the scene.

        Parameters:
            name (str): Name of the target

        Returns:
            The target (Target)
        """
        return Target(self.viewer.getTarget(name))


    def addArrow(self, name, origin, direction, length=1, color='#ffff00', shading=False,
                 headLength=None, headWidth=None, radius=None):
        """Add an arrow to the scene.

        Parameters:
            name (str): Name of the arrow
            origin (list/NumPy array): Point at which the arrow starts
            direction (list/NumPy array): Direction from origin (must be a unit vector)
            length (float): Length of the arrow (default is 1)
            color (int/str): Color of the arrow (by default: 0xffff00)
            shading (bool): Indicates if the arrow must be affected by lights (by default: False)
            headLength (float): The length of the head of the arrow (default is 0.2 * length)
            headWidth (float): The width of the head of the arrow (default is 0.2 * headLength)
            radius (float): The radius of the line part of the arrow (default is 0.1 * headWidth)
        """
        if isinstance(origin, np.ndarray):
            origin = list(origin)

        if isinstance(direction, np.ndarray):
            direction = list(direction)

        headLength = headLength if headLength is not None else length * 0.2
        headWidth = headWidth if headWidth is not None else headLength * 0.2
        radius = radius if radius is not None else headWidth * 0.1

        return Arrow(
            self.viewer.addArrow(
                name,
                three.Vector3.new(*origin),
                three.Vector3.new(*direction),
                length,
                color,
                shading,
                headLength,
                headWidth,
                radius
            )
        )


    def removeArrow(self, name):
        """Remove an arrow from the scene.

        Parameters:
            name (str): Name of the arrow
        """
        self.viewer.removeTarget(name)


    def getArrow(self, name):
        """Returns an arrow from the scene.

        Parameters:
            name (str): Name of the arrow

        Returns:
            The arrow (Arrow)
        """
        return Arrow(self.viewer.getArrow(name))


    def addPath(self, name, points, radius=0.01, color='#ffff00', shading=False,
                transparent=False, opacity=0.5):
        """Add a path to the scene.

        Parameters:
            name (str): Name of the path
            points (2D NumPy array/list of lists of 3 floats): Points defining the path
            radius (float): The radius of the path (default is 0.01)
            color (int/str): Color of the path (by default: 0xffff00)
            shading (bool): Indicates if the path must be affected by lights (by default: False)
            transparent (bool): Indicates if the path must be transparent (by default: False)
            opacity (float): Opacity level for transparent paths (between 0 and 1, default: 0.5)
        """
        if isinstance(points, np.ndarray):
            points = [ list(x) for x in list(points) ]

        return Path(
            self.viewer.addPath(
                name,
                to_js(points),
                radius,
                color,
                shading,
                transparent,
                opacity
            )
        )


    def removePath(self, name):
        """Remove a path from the scene.

        Parameters:
            name (str): Name of the path
        """
        self.viewer.removePath(name)


    def getPath(self, name):
        """Returns a path from the scene.

        Parameters:
            name (str): Name of the path

        Returns:
            The path (Path)
        """
        return Path(self.viewer.getPath(name))


    def addPoint(self, name, position, radius=0.01, color='#ffff00', label=None, shading=False,
                 transparent=False, opacity=0.5):
        """Add a point to the scene.

        Parameters:
            name (str): Name of the point
            position (list/NumPy array): Position of the point
            radius (float): The radius of the point (default is 0.01)
            color (int/str): Color of the point (by default: 0xffff00)
            label (str): LaTeX text to display near the point (by default: None)
            shading (bool): Indicates if the point must be affected by lights (by default: False)
            transparent (bool): Indicates if the point must be transparent (by default: False)
            opacity (float): Opacity level for transparent points (between 0 and 1, default: 0.5)
        """
        if isinstance(position, np.ndarray):
            position = list(position)

        return Point(
            self.viewer.addPoint(
                name,
                three.Vector3.new(*position),
                radius,
                color,
                label,
                shading,
                transparent,
                opacity
            )
        )


    def removePoint(self, name):
        """Remove a point from the scene.

        Parameters:
            name (str): Name of the point
        """
        self.viewer.removePoint(name)


    def getPoint(self, name):
        """Returns a point from the scene.

        Parameters:
            name (str): Name of the point

        Returns:
            The path (Path)
        """
        return Point(self.viewer.getPoint(name))


    def addGaussian(self, name, mu, sigma, color='#ffff00', listener=None):
        """Add a gaussian to the scene

        Parameters:
            name (str): Name of the gaussian
            mu (list/NumPy array): Position of the gaussian
            sigma (list of lists/NumPy array, 3x3): Covariance matrix of the gaussian
            color (int/str): Color of the gaussian (by default: 0xffff00)
            listener (function): Function to call when the gaussian is modified using the mouse
        """
        def _listener(gaussianjs, dragging):
            listener(Gaussian(gaussianjs), dragging)

        if isinstance(mu, np.ndarray):
            mu = list(mu)

        if isinstance(sigma, list):
            sigma = np.array(sigma)

        return Gaussian(
            self.viewer.addGaussian(
                name,
                three.Vector3.new(*mu),
                three.Matrix3.new().set(*list(sigma.flatten())),
                color,
                create_proxy(_listener) if listener is not None else None,
            )
        )


    def removeGaussian(self, name):
        """Remove a gaussian from the scene.

        Parameters:
            name (str): Name of the gaussian
        """
        self.viewer.removeGaussian(name)


    def getGaussian(self, name):
        """Returns a gaussian from the scene.

        Parameters:
            name (str): Name of the gaussian

        Returns:
            The gaussian (Gaussian)
        """
        return Gaussian(self.viewer.getGaussian(name))


    def getPhysicalBody(self, name):
        """Retrieve a body from the physics simulation.

        Parameters:
            name (str): Name of the body

        Returns:
            The body (PhysicalBody)
        """
        body = self.viewer.getPhysicalBody(name)
        if body is None:
            return None

        return PhysicalBody(body)


    def translateCamera(self, delta):
        if isinstance(delta, np.ndarray):
            delta = list(delta)

        self.viewer.translateCamera(three.Vector3.new(*delta))


    def enableLogmap(self, robot, target, position='left', size=None):
        if isinstance(robot, Robot):
            robot = robot.name

        if isinstance(target, Target):
            target = target.name

        self.viewer.enableLogmap(robot, target, position, size)


    def disableLogmap(self):
        self.viewer.disableLogmap()


    @property
    def logmapTarget(self):
        """Returns the name of the target used by the logmap visualisation"""
        if self.viewer.logmap is not None:
            return self.viewer.logmap.targetName


    @logmapTarget.setter
    def logmapTarget(self, name):
        """Sets the name of the target to be used by the logmap visualisation"""
        if self.viewer.logmap is not None:
            self.viewer.logmap.targetName = to_js(name)


    def render(self):
        self.viewer.render()


    async def stop(self):
        await self.viewer.stop()



class Robot:
    """The robot"""

    def __init__(self, robotjs, viewerjs):
        """Constructor, for internal use only"""
        self.robot = robotjs
        self.viewer = viewerjs


    @property
    def name(self):
        """Returns the name of the object"""
        return self.robot.name


    @property
    def jointPositions(self):
        """Returns the position of the joints of the robot (as a NumPy array)"""
        return np.array(self.robot.getJointPositions().to_py())


    @jointPositions.setter
    def jointPositions(self, positions):
        """Sets the position of the joints of the robot

        Parameters:
            positions (list/NumpPy array): the joint positions
        """
        if isinstance(positions, np.ndarray):
            positions = list(positions)

        self.robot.setJointPositions(to_js(positions))


    @property
    def control(self):
        """Returns the position of the joints of the robot (as a NumPy array)"""
        return np.array(self.robot.getControl().to_py())


    @control.setter
    def control(self, control):
        """Sets the position of the joints of the robot

        Parameters:
            positions (list/NumpPy array): the joint positions
        """
        if isinstance(control, np.ndarray):
            control = list(control)

        self.robot.setControl(to_js(control))


    @property
    def defaultPose(self):
        return np.array(self.robot.getDefaultPose().to_py())


    def applyDefaultPose(self):
        self.robot.applyDefaultPose(self.defaultPose)


    @property
    def endEffectorPosition(self):
        pos = self.robot.getEndEffectorPosition().to_py()
        return np.array([pos.x, pos.y, pos.z])


    @property
    def endEffectorOrientation(self):
        quat = self.robot.getEndEffectorOrientation().to_py()
        return np.array([quat.x, quat.y, quat.z, quat.w])


    @property
    def endEffectorTransforms(self):
        """Returns the position and orientation of the end-effector of the robot in
        a Numpy array of the form: [px, py, pz, qx, qy, qz, qw]

        Returns:
            A NumPy array like [px, py, pz, qx, qy, qz, qw]
        """
        return np.array(self.robot.getEndEffectorTransforms().to_py())


    @property
    def endEffectorDesiredTransforms(self):
        """Returns the desired position and orientation for the end-effector of the robot
        in a Numpy array of the form: [px, py, pz, qx, qy, qz, qw]

        The desired position and orientation are those of the manipulator of the
        end-effector (if enabled, see 'Viewer3D.endEffectorManipulation'), that the
        user can move freely.

        Returns:
            A NumPy array like [px, py, pz, qx, qy, qz, qw]
        """
        return np.array(self.robot.getEndEffectorDesiredTransforms().to_py())


    @property
    def toolEnabled(self):
        return self.robot.isToolEnabled()


    @toolEnabled.setter
    def toolEnabled(self, enabled):
        self.robot.enableTool(enabled)


    def fkin(self, positions, offset=None):
        """Forward kinematics computation given some joint positions

        Parameters:
            positions (list/NumpPy array): the joint positions. If a 2D Numpy array is
                                           provided, one forward kinematics computation
                                           is performed for each column

        Returns:
            A Numpy array containing the position and orientation of the end-effector
            [px, py, pz, qx, qy, qz, qw] if the robot has the specified joint positions.

            If 'positions' is a 2D Numpy array, the result is also a 2D array, with each
            column corresponding to a column in 'positions'.
        """
        if offset is not None:
            offset = three.Vector3.new(*offset)

        if isinstance(positions, np.ndarray):
            if (len(positions.shape) == 2) and (positions.shape[1] > 1):
                result = np.ndarray((7, positions.shape[1]))

                for i in range(positions.shape[1]):
                    result[:, i] = self.robot.fkin(to_js(list(positions[:, i])), offset).to_py()

                return result

            else:
                positions = list(positions)

        return np.array(self.robot.fkin(to_js(positions), offset).to_py())


    def ik(self, mu, nbJoints=None, offset=None, limit=None, damping=False):
        x = self.control
        startx = x.copy()

        indices = slice(0, 7)
        if len(mu) == 3:
            indices = slice(0, 3)
        elif len(mu) == 4:
            indices = slice(3, 7)

        if nbJoints is None:
            nbJoints = len(x)

        if not isinstance(mu, np.ndarray):
            mu = np.array(mu)

        damping = damping or (nbJoints < len(x))

        done = False
        i = 0
        while not(done) and ((limit is None) or (i < limit)):
            f = self.fkin(x[:nbJoints], offset)

            if len(mu) == 3:
                diff = mu - f[indices]
            elif len(mu) == 4:
                diff = logmap_S3(mu, f[indices])
            else:
                diff = logmap(mu, f)

            J = self.Jkin(x[:nbJoints], offset)
            J = J[indices, :]

            if damping:
                pinvJ = np.linalg.inv(J.T @ J + np.eye(nbJoints) * 1e-2) @ J.T # Damped pseudoinverse
            else:
                pinvJ = np.linalg.pinv(J)

            u = pinvJ @ diff

            x[:nbJoints] += u * 0.1

            i += 1

            if np.linalg.norm(x - startx) < 1e-5:
                done = True

        control = self.control
        control[:nbJoints] = x[:nbJoints]
        self.control = control

        return done


    def Jkin(self, positions, offset=None):
        """Jacobian with numerical computation, on a subset of the joints
        """
        eps = 1e-6
        D = len(positions)

        # Matrix computation
        X = np.tile(positions.reshape((D,1)), [1,D])
        F1 = self.fkin(X, offset)
        F2 = self.fkin(X + np.identity(D) * eps, offset)
        J = logmap(F2, F1) / eps

        if len(J.shape) == 1:
            J = J.reshape((-1,1))

        return J


    @property
    def isGripperOpen(self):
        return self.robot.isGripperOpen()


    @property
    def isGripperClosed(self):
        return self.robot.isGripperClosed()

    @property
    def isGripperHoldingSomeObject(self):
        return self.robot.isGripperHoldingSomeObject()

    @property
    def gripperAbduction(self):
        """Returns the current abduction of the gripper, between 0.0 (closed) and 1.0 (fully open)"""
        return self.robot.getGripperAbduction()


    def openGripper(self):
        """Opens the gripper (will take some time to complete)"""
        return self.robot.openGripper()


    def closeGripper(self):
        """Closes the gripper (will take some time to complete)"""
        return self.robot.closeGripper()



class Object3D:
    """Base class for all the objects that can be placed in the scene
    """

    def __init__(self, objectjs):
        """Constructor, for internal use only"""
        self.object = objectjs


    @property
    def name(self):
        """Returns the name of the object"""
        return self.object.name


    @property
    def position(self):
        """Returns the position of the object (as a NumPy array)"""
        return np.array([
            self.object.position.x,
            self.object.position.y,
            self.object.position.z,
        ])


    @position.setter
    def position(self, position):
        """Sets the position of the object

        Parameters:
            position (list/NumpPy array): the desired object position
        """
        self.object.position.set(position[0], position[1], position[2])


    @property
    def orientation(self):
        """Returns the orientation (x, y, z, w) of the object (as a NumPy array)"""
        return np.array([
            self.object.quaternion.x,
            self.object.quaternion.y,
            self.object.quaternion.z,
            self.object.quaternion.w,
        ])


    @orientation.setter
    def orientation(self, orientation):
        """Sets the orientation of the object

        Parameters:
            orientation (list/NumpPy array): the desired object orientation (x, y, z, w)
        """
        self.object.quaternion.set(orientation[0], orientation[1], orientation[2], orientation[3])


    @property
    def transforms(self):
        """Returns the position and orientation of the object in a Numpy array of the form:
        [px, py, pz, qx, qy, qz, qw]"""
        return np.array(self.object.transforms().to_py())



class Target(Object3D):
    """Represents a target, an object that can be manipulated by the user that can for example
    be used to define a destination position and orientation for the end-effector of the robot.
    """

    def __init__(self, targetjs):
        """Constructor, for internal use only"""
        super().__init__(targetjs)



class Arrow(Object3D):
    """An arrow, that can be placed in the scene
    """

    def __init__(self, arrowjs):
        """Constructor, for internal use only"""
        super().__init__(arrowjs)


    def setColor(self, color):
        """Sets the color of the arrow

        Parameters:
            color (int/str): Color of the arrow
        """
        self.object.setColor(color)


    def setLength(self, length, headLength=None, headWidth=None, radius=None):
        """Sets the orientation of the target

        Parameters:
            length (float): The desired length
            headLength (float): The length of the head of the arrow (default is 0.2 * length)
            headWidth (float): The width of the head of the arrow (default is 0.2 * headLength)
            radius (Number): The radius of the line part of the arrow (default is 0.1 * headWidth)
        """
        headLength = headLength if headLength is not None else length * 0.2
        headWidth = headWidth if headWidth is not None else headLength * 0.2
        radius = radius if radius is not None else headWidth * 0.1
        self.object.setLength(length, headLength, headWidth, radius)


    @property
    def origin(self):
        """Returns the position of the target (as a NumPy array)"""
        return np.array([
            self.object.position.x,
            self.object.position.y,
            self.object.position.z,
        ])


    @origin.setter
    def origin(self, origin):
        """Sets the point at which the arrow starts

        Parameters:
            origin (list/NumpPy array): the desired point
        """
        self.object.position.set(origin[0], origin[1], origin[2])


    @property
    def direction(self):
        """Returns the direction of the arrow (as a NumPy array)"""
        direction = three.Vector3.new(0, 0, 1);
        direction.applyQuaternion(self.object.quaternion);

        return np.array([
            direction.x,
            direction.y,
            direction.z,
        ])


    @direction.setter
    def direction(self, direction):
        """Sets the direction of the arrow

        Parameters:
            direction (list/NumpPy array): the desired direction
        """
        if isinstance(direction, np.ndarray):
            direction = list(direction)

        self.object.setDirection(three.Vector3.new(*direction))



class Path(Object3D):
    """A path, that can be placed in the scene
    """

    def __init__(self, pathjs):
        """Constructor, for internal use only"""
        super().__init__(pathjs)



class Point(Object3D):
    """A point, with an optional label, that can be placed in the scene
    """

    def __init__(self, pointjs):
        """Constructor, for internal use only"""
        super().__init__(pointjs)



class Gaussian(Object3D):
    """A gaussian, that can be placed in the scene
    """

    def __init__(self, gaussianjs):
        """Constructor, for internal use only"""
        super().__init__(gaussianjs)


    def setColor(self, color):
        """Sets the color of the gaussian

        Parameters:
            color (int/str): Color of the gaussian
        """
        self.object.setColor(color)


    @property
    def sigma(self):
        """Returns the covariance matrix of the gaussian (as a NumPy array, 3x3)"""
        sigma = self.object.sigma()
        return np.array(sigma.elements.to_py()).reshape((3, 3)).T


    @sigma.setter
    def sigma(self, sigma):
        """Sets the covariance matrix of the gaussian

        Parameters:
            sigma (list of lists/NumpPy array, 3x3): the desired covariance matrix
        """
        if isinstance(sigma, list):
            sigma = np.array(sigma)

        self.object.setSigma(three.Matrix3.new().set(*list(sigma.flatten())))


class PhysicalBody:

    def __init__(self, bodyjs):
        """Constructor, for internal use only"""
        self.body = bodyjs


    @property
    def name(self):
        """Returns the name of the body"""
        return self.body.name


    @property
    def position(self):
        """Returns the position of the body (as a NumPy array)"""
        pos = self.body.position()
        return np.array([pos.x, pos.y, pos.z])



def QuatMatrix(q):
    """Matrix form of quaternion

    Note: quaternions elements are ordered as [x, y, z, w]
    """
    return np.array([
        [q[3], -q[2], q[1], q[0]],
        [q[2], q[3], -q[0], q[1]],
        [-q[1], q[0], q[3], q[2]],
        [-q[0], -q[1], -q[2], q[3]],
    ])



def acoslog(x):
    """Arcosine redefinition to make sure the distance between antipodal quaternions is zero
    """
    try:
        y = math.acos(x)
    except ValueError:
        return math.nan

    if x < 0:
        y = y - np.pi
    return y



def logmap_S3(x, x0):
    """Logarithmic map for S^3 manifold (with e in tangent space)
    """
    R = QuatMatrix(x0)
    x = R.T @ x
    sc = acoslog(x[3]) / np.sqrt(1.0 - x[3]**2)
    if math.isnan(sc):
        sc = 1.0
    return x[:3] * sc



def logmap(f, f0):
    """Logarithmic map for R^3 x S^3 manifold (with e in tangent space)
    """
    if len(f.shape) == 1:
        e = np.ndarray((6,))
        e[0:3] = (f[0:3] - f0[0:3])
        e[3:] = logmap_S3(f[3:], f0[3:])
    else:
        e = np.ndarray((6, f.shape[1]))
        e[0:3,:] = (f[0:3,:] - f0[0:3,:])
        for t in range(f.shape[1]):
            e[3:,t] = logmap_S3(f[3:,t], f0[3:,t])
    return e



def sigmaFromQuaternionAndScale(quaternion, scale):
    """Computes the covariance matrix of a gaussian from an orientation and scale

    Parameters:
        quaternion (list/NumpPy array/Quaternion): the orientation
        scale (list/NumpPy array): the scale

    Returns:
        A 3x3 NumpPy array
    """
    if isinstance(quaternion, np.ndarray):
        quaternion = list(quaternion)

    if isinstance(quaternion, list):
        quaternion = three.Quaternion.new(*quaternion)

    if isinstance(scale, np.ndarray):
        scale = list(scale)

    sigma = gaussians.sigmaFromQuaternionAndScale(quaternion, three.Vector3.new(*scale))

    return np.array(sigma.elements.to_py()).reshape((3, 3)).T



def sigmaFromMatrix(matrix):
    """Computes the covariance matrix of a gaussian from a rotation and scaling matrix (either
    3x3 or 4x4, in which case the upper 3x3 part is used)

    Parameters:
        matrix (list of lists/NumpPy array, 3x3 or 4x4): the matrix

    Returns:
        A 3x3 NumpPy array
    """
    if isinstance(matrix, list):
        matrix = np.array(matrix)

    sigma = gaussians.sigmaFromMatrix3(three.Matrix3.new(*list(matrix[:3,:3].flatten())))

    return np.array(sigma.elements.to_py()).reshape((3, 3)).T



def matrixFromSigma(sigma):
    """Computes the rotation and scaling matrix corresponding to the covariance matrix of a gaussian

    Parameters:
        sigma (list of lists/NumpPy array, 3x3): the covariance matrix

    Returns:
        A 3x3 NumpPy array
    """
    if isinstance(sigma, list):
        sigma = np.array(sigma)

    matrix = gaussians.matrixFromSigma(three.Matrix3.new(*list(sigma.flatten())))

    return np.array(matrix.elements.to_py()).reshape((3, 3)).T
