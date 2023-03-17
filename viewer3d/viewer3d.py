#
# SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
#
# SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
#
# SPDX-License-Identifier: MIT
#

from js import Viewer3Djs
from js import three
from pyodide.ffi import create_proxy
from pyodide.ffi import to_js
import numpy as np


class Viewer3D:
    """Python entry point for the 'viewer3d.js' library, used to display and interact with a 3D
    representation of the Panda robotic arm.

    This class mainly acts as a bridge between the Python and JavaScript worlds, handling all the
    needed type conversions.
    """

    def __init__(self, domElement=None, parameters=None, onready=None):
        """Constructs the 3D viewer

        Parameters:
            domElement (element): The DOM element to use for displaying the robotic arm
            parameters (dict): Additional optional parameters, to customize the behavior of the
                               3D viewer (see below)
            onready (function): Function to call once all assets are loaded and the scene is ready

        If no DOM element is provided, one is created. It is the duty of the caller to insert it
        somewhere in the DOM (see 'Viewer3D.domElement').

        Optional parameters:
            logmap_sphere (bool):
                enable the display of the logmap between the orientation of the end-effector and
                the one of a target (default: false)

            logmap_sphere_size (int)
                approximate size in pixels of the logmap sphere (default: 1/10 of the DOM
                element width at creation)

            shadows (bool):
                enable the rendering of the shadows (default: true)

            show_joint_positions (bool):
                enable the display of an visual indicator around each joint (default: false)

            statistics (bool):
                enable the display of statistics about the rendering performance (default: false)

            theme (Themes):
                the theme to use (default: Theme.Default)
        """
        self.viewer = Viewer3Djs.new(
            domElement,
            to_js(parameters),
            create_proxy(onready) if onready is not None else None
        )

        self.robotpy = None


    @property
    def domElement(self):
        """Returns the DOM element containing the 3D viewer"""
        return self.viewer.domElement


    @property
    def robot(self):
        """Returns the robot

        Returns:
            The robot (Robot)
        """
        if self.robotpy is None:
            self.robotpy = Robot(self.viewer.robot, self.viewer.fkrobot)

        return self.robotpy


    def setRenderingCallback(self, callback):
        """Register a function that should be called once per frame.

        This callback function can for example be used to update the positions of the joints.

        The signature of the callback function is: callback(delta), with 'delta' the time elapsed
        since the last frame, in seconds.

        Note that only one function can be registered at a time. If 'callback' is 'None', no
        function is called anymore.
        """
        if callback is not None:
            self.viewer.setRenderingCallback(create_proxy(callback))
        else:
            self.viewer.setRenderingCallback(None)


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


    def addTarget(self, name, position, orientation, color=None):
        """Add a target to the scene, an object that can be manipulated by the user that can
        be used to define a destination position and orientation for the end-effector of the
        robot.

        Parameters:
            name (str): Name of the target
            position (list/NumPy array): The position (x, y, z) of the target
            orientation (list/NumPy array): The orientation (x, y, z, w) of the target
            color (int/str): Color of the target (by default: 0x0000aa)
        """
        if isinstance(position, np.ndarray):
            position = list(position)

        if isinstance(orientation, np.ndarray):
            orientation = list(orientation)

        return Target(
            self.viewer.addTarget(
                name,
                three.Vector3.new(*position),
                three.Quaternion.new(*orientation),
                color
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


    def addArrow(self, name, origin, direction, length=None, color=None, headLength=None, headWidth=None):
        """Add an arrow to the scene.

        Parameters:
            name (str): Name of the arrow
            origin (list/NumPy array): Point at which the arrow starts
            direction (list/NumPy array): Direction from origin (must be a unit vector)
            length (float): Length of the arrow (default is 1)
            color (int/str): Color of the arrow (by default: 0xffff00)
            headLength (float): The length of the head of the arrow (default is 0.2 * length)
            headWidth (float): The width of the head of the arrow (default is 0.2 * headLength)
        """
        if isinstance(origin, np.ndarray):
            origin = list(origin)

        if isinstance(direction, np.ndarray):
            direction = list(direction)

        return Arrow(
            self.viewer.addArrow(
                name,
                three.Vector3.new(*origin),
                three.Vector3.new(*direction),
                length,
                color,
                headLength,
                headWidth
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
        return np.array(self.viewer.getEndEffectorDesiredTransforms().to_py())


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



class Robot:
    """The robot"""

    def __init__(self, robotjs, fkrobotjs):
        """Constructor, for internal use only"""
        self.robot = robotjs
        self.fkrobot = fkrobotjs


    @property
    def jointPositions(self):
        """Returns the position of the joints of the robot (as a NumPy array)"""
        return np.array(self.robot.getPose().to_py())


    @jointPositions.setter
    def jointPositions(self, positions):
        """Sets the position of the joints of the robot

        Parameters:
            positions (list/NumpPy array): the joint positions
        """
        if isinstance(positions, np.ndarray):
            positions = list(positions)

        self.robot.setPose(to_js(positions))


    @property
    def endEffectorTransforms(self):
        """Returns the position and orientation of the end-effector of the robot in
        a Numpy array of the form: [px, py, pz, qx, qy, qz, qw]

        Returns:
            A NumPy array like [px, py, pz, qx, qy, qz, qw]
        """
        return np.array(self.robot.getEndEffectorTransforms().to_py())


    def fkin(self, positions):
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
        if isinstance(positions, np.ndarray):
            if (len(positions.shape) == 2) and (positions.shape[1] > 1):
                result = np.ndarray(positions.shape)
                for i in range(positions.shape[1]):
                    p, q = self.fkrobot.fkin(to_js(list(positions[:, i])))
                    result[:, i] = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
                return result

            positions = list(positions)

        p, q = self.fkrobot.fkin(to_js(positions))
        return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])


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



class Target:
    """Represents a target, an object that can be manipulated by the user that can for example
    be used to define a destination position and orientation for the end-effector of the robot.
    """

    def __init__(self, targetjs):
        """Constructor, for internal use only"""
        self.target = targetjs


    @property
    def name(self):
        """Returns the name of the target"""
        return self.target.name


    @property
    def position(self):
        """Returns the position of the target (as a NumPy array)"""
        return np.array([
            self.target.position.x,
            self.target.position.y,
            self.target.position.z,
        ])


    @position.setter
    def position(self, position):
        """Sets the position of the target

        Parameters:
            position (list/NumpPy array): the desired target position
        """
        self.target.position.set(position[0], position[1], position[2])


    @property
    def orientation(self):
        """Returns the orientation (x, y, z, w) of the target (as a NumPy array)"""
        return np.array([
            self.target.quaternion.x,
            self.target.quaternion.y,
            self.target.quaternion.z,
            self.target.quaternion.w,
        ])


    @orientation.setter
    def orientation(self, orientation):
        """Sets the orientation of the target

        Parameters:
            orientation (list/NumpPy array): the desired target orientation (x, y, z, w)
        """
        self.target.quaternion.set(orientation[0], orientation[1], orientation[2], orientation[3])


    @property
    def transforms(self):
        """Returns the position and orientation of the target in a Numpy array of the form:
        [px, py, pz, qx, qy, qz, qw]"""
        return np.array(self.target.transforms().to_py())



class Arrow:
    """An arrow, that can be placed in the scene
    """

    def __init__(self, arrowjs):
        """Constructor, for internal use only"""
        self.arrow = arrowjs


    @property
    def name(self):
        """Returns the name of the arrow"""
        return self.arrow.name


    def setColor(self, color):
        """Sets the color of the arrow

        Parameters:
            color (int/str): Color of the arrow
        """
        self.arrow.setColor(color)


    def setLength(self, length, headLength=None, headWidth=None):
        """Sets the orientation of the target

        Parameters:
            length (float): The desired length
            headLength (float): The length of the head of the arrow (default is 0.2 * length)
            headWidth (float): The width of the head of the arrow (default is 0.2 * headLength)
        """
        self.arrow.setLength(length, headLength, headWidth)


    @property
    def origin(self):
        """Returns the position of the target (as a NumPy array)"""
        return np.array([
            self.arrow.position.x,
            self.arrow.position.y,
            self.arrow.position.z,
        ])


    @origin.setter
    def origin(self, origin):
        """Sets the point at which the arrow starts

        Parameters:
            origin (list/NumpPy array): the desired point
        """
        self.arrow.position.set(origin[0], origin[1], origin[2])


    @property
    def direction(self):
        """Returns the position of the target (as a NumPy array)"""
        direction = three.Vector3.new(0, 0, 1);
        direction.applyQuaternion(self.arrow.quaternion);

        return np.array([
            direction.x,
            direction.y,
            direction.z,
            direction.w,
        ])


    @direction.setter
    def direction(self, direction):
        """Returns the position and orientation of the target in a Numpy array of the form:
        [px, py, pz, qx, qy, qz, qw]"""
        if isinstance(direction, np.ndarray):
            direction = list(direction)

        self.arrow.setDirection(three.Vector3.new(*direction))
