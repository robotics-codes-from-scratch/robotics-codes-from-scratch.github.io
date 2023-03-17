/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import * as THREE from 'three';
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import Stats from 'three/examples/jsm/libs/stats.module.js';
import { TWEEN } from 'three/examples/jsm/libs/tween.module.min.js'
import { CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';

import { loadRobot } from './loader.js';
import TransformControlsManager from './transformcontrols.js';
import FKRobot from './robots/fkrobot.js';
import Logmap from './logmap.js';
import TargetList from './helpers/targetlist.js';
import ArrowList from './helpers/arrowlist.js';


const Themes = Object.freeze({
    Default: Symbol("default"),
    Simple: Symbol("simple")
});



const ThemeParameters = new Map([
    [Themes.Default, {
        backgroundColor: new THREE.Color(0x363b4b),
        hemisphereLightColor: 0xffeeee,
    }],
    [Themes.Simple, {
        backgroundColor: new THREE.Color(0xffffff),
        hemisphereLightColor: 0xaa99999,
    }]
])


const InteractionStates = Object.freeze({
    Default: Symbol("default"),
    Manipulation: Symbol("manipulation"),
    JointHovering: Symbol("joint_hovering"),
    JointDisplacement: Symbol("joint_displacement")
});



/* Entry point for the 'viewer3d.js' library, used to display and interact with a 3D
representation of the Panda robotic arm.
*/
class Viewer3D {

    /* Constructs the 3D viewer

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
    */
    constructor(domElement, parameters, onready) {
        this.parameters = this._checkParameters(parameters)

        this.domElement = domElement;

        if (this.domElement == undefined)
            this.domElement = document.createElement('div');

        if (!this.domElement.classList.contains('viewer3d'))
            this.domElement.classList.add('viewer3d');

        this.camera = null;
        this.scene = null;
        this.robot = null;
        this.fkrobot = null;

        this.tcpTarget = null;
        this.targets = new TargetList();
        this.arrows = new ArrowList();

        this.interactionState = InteractionStates.Default;

        this.hoveredGroup = null;
        this.hoveredJoint = null;
        this.previousPointer = null;

        this.renderer = null;
        this.labelRenderer = null;
        this.clock = new THREE.Clock();

        this.cameraControl = null;
        this.transformControls = null;
        this.raycaster = new THREE.Raycaster();
        this.stats = null;

        this.logmap = null;

        this.renderingCallback = null;

        this.controlsEnabled = true;
        this.endEffectorManipulationEnabled = false;
        this.jointsManipulationEnabled = true;

        this._initScene();

        if (this.parameters.get('logmap_sphere')) {
            this.logmap = new Logmap(
                this.domElement,
                ThemeParameters.get(this.parameters.get('theme')).backgroundColor,
                this.parameters.get('logmap_sphere_size')
            );
            this.scene.background = null;
        }

        this._render();

        loadRobot()
            .then(robot => {
                this.robot = robot;
                this.scene.add(this.robot.model);

                if (this.parameters.get('show_joint_positions'))
                    this.robot.createJointPositionHelpers();

                if (this.endEffectorManipulationEnabled)
                    this._createTcpTarget();

                this.fkrobot = new FKRobot(this.robot);

                if (onready != null)
                    onready();
        })
    }


    /* Register a function that should be called once per frame.

    This callback function can for example be used to update the positions of the joints.

    The signature of the callback function is: callback(delta), with 'delta' the time elapsed
    since the last frame, in seconds.

    Note that only one function can be registered at a time. If 'callback' is 'null', no
    function is called anymore.
    */
    setRenderingCallback(renderingCallback) {
        this.renderingCallback = renderingCallback;
    }


    /* Enables/disables the manipulation controls

    Manipulation controls include the end-effector and the target manipulators.
    */
    enableControls(enabled) {
        this.controlsEnabled = enabled;
        
        this.transformControls.enable(enabled);

        if (enabled && (this.tcpTarget != null)) {
            this.robot.tcp.object.getWorldPosition(this.tcpTarget.position);
            this.robot.tcp.object.getWorldQuaternion(this.tcpTarget.quaternion);
        }
    }


    /* Indicates if the manipulation controls are enabled

    Manipulation controls include the end-effector and the target manipulators.
    */
    areControlsEnabled() {
        return this.this.controlsEnabled;
    }


    /* Enables/disables the manipulation of the end effector (when the user clicks on it)

    Note that if 'Viewer3D.controlsEnabled' is 'false', the end-effector can't be
    manipulated regardless of the value of this property.
    */
    enableEndEffectorManipulation(enabled) {
        this.endEffectorManipulationEnabled = enabled;

        if (enabled && (this.tcpTarget == null))
            this._createTcpTarget();
    }


    /* Indicates if the manipulation of the end effector is enabled (when the user clicks
    on it).

    Note that if 'Viewer3D.controlsEnabled' is 'false', the end-effector can't be
    manipulated regardless of the value of this property.
    */
    isEndEffectorManipulationEnabled() {
        return this.endEffectorManipulationEnabled;
    }


    /* Enables/disables the manipulation of the joint positions (when the user clicks on
    them or use the mouse wheel).

    Note that if 'Viewer3D.controlsEnabled' is 'false', the position of the joints
    can't be changed using the mouse regardless of the value of this property.
    */
    enableJointsManipulation(enabled) {
        this.jointsManipulationEnabled = enabled;

        if ((this.interactionState == InteractionStates.JointHovering) ||
            (this.interactionState == InteractionStates.JointDisplacement)) {
                this._switchToInteractionState(InteractionStates.Default);
        }
    }


    /* Indicates if the manipulation of the joint positions is enabled (when the user
    clicks on them or use the mouse wheel).

    Note that if 'Viewer3D.controlsEnabled' is 'false', the position of the joints
    can't be changed using the mouse regardless of the value of this property.
    */
    isJointsManipulationEnabled() {
        return this.jointsManipulationEnabled;
    }


    /* Add a target to the scene, an object that can be manipulated by the user that can
    be used to define a destination position and orientation for the end-effector of the
    robot.

    Parameters:
        name (str): Name of the target
        position (Vector3): The position of the target
        orientation (Quaternion): The orientation of the target
        color (int/str): Color of the target (by default: 0x0000aa)
    */
    addTarget(name, position, orientation, color) {
        const target = this.targets.create(name, position, orientation, color);
        this.scene.add(target);
        return target;
    }


    /* Remove a target from the scene.

    Parameters:
        name (str): Name of the target
    */
    removeTarget(name) {
        this.targets.destroy(name);
    }


    /* Returns a target from the scene.

    Parameters:
        name (str): Name of the target
    */
    getTarget(name) {
        return this.targets.get(name);
    }


    /* Add an arrow to the scene

    Parameters:
        name (str): Name of the arrow
        origin (Vector3): Point at which the arrow starts
        direction (Vector3): Direction from origin (must be a unit vector)
        length (Number): Length of the arrow (default is 1)
        color (int/str): Color of the arrow (by default: 0xffff00)
        headLength (Number): The length of the head of the arrow (default is 0.2 * length)
        headWidth (Number): The width of the head of the arrow (default is 0.2 * headLength)
    */
    addArrow(name, origin, direction, length, color, headLength, headWidth) {
        const arrow = this.arrows.create(name, origin, direction, length, color, headLength, headWidth);
        this.scene.add(arrow);
        return arrow;
    }


    /* Remove an arrow from the scene.

    Parameters:
        name (str): Name of the arrow
    */
    removeArrow(name) {
        this.arrows.destroy(name);
    }


    /* Returns an arrow from the scene.

    Parameters:
        name (str): Name of the arrow
    */
    getArrow(name) {
        return this.arrows.get(name);
    }


    /* Returns the desired position and orientation for the end-effector of the robot in an
    array of the form: [px, py, pz, qx, qy, qz, qw]

    The desired position and orientation are those of the manipulator of the
    end-effector (if enabled, see 'Viewer3D.endEffectorManipulation'), that the
    user can move freely.

    Returns:
        [px, py, pz, qx, qy, qz, qw]
    */
    getEndEffectorDesiredTransforms() {
        if (this.endEffectorManipulationEnabled) {
            return [
                this.tcpTarget.position.x, this.tcpTarget.position.y, this.tcpTarget.position.z,
                this.tcpTarget.quaternion.x, this.tcpTarget.quaternion.y, this.tcpTarget.quaternion.z, this.tcpTarget.quaternion.w,
            ];
        }

        if (this.robot != null)
            return this.robot.getEndEffectorTransforms();

        return [0, 0, 0, 0, 0, 0, 1];
    }


    _checkParameters(parameters) {
        if (parameters == null)
            parameters = new Map();
        else if (!(parameters instanceof Map))
            parameters = new Map(Object.entries(parameters));

        const defaults = new Map([
            ['logmap_sphere', false],
            ['logmap_sphere_size', null],
            ['shadows', true],
            ['show_joint_positions', false],
            ['statistics', false],
            ['theme', Themes.Default],
        ]);

        return new Map([...defaults, ...parameters]);
    }


    _initScene() {
        const theme = ThemeParameters.get(this.parameters.get('theme'));

        // In ROS, the Z-axis of the models points upwards
        THREE.Object3D.DefaultUp = new THREE.Vector3(0, 0, 1);

        // Statistics
        if (this.parameters.get('statistics')) {
            this.stats = new Stats();
            this.stats.dom.classList.add('statistics');
            this.stats.dom.style.removeProperty('position');
            this.stats.dom.style.removeProperty('top');
            this.stats.dom.style.removeProperty('left');
            this.domElement.appendChild(this.stats.dom);
        }

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, this.domElement.clientWidth / this.domElement.clientHeight, 0.1, 50);
        this.camera.position.set(1, 2, 1);
        this.camera.lookAt(0, 0, 0.5);

        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = theme.backgroundColor.clone();

        // Floor
        if (this.parameters.get('theme') == Themes.Default) {
            // Grid
            const grid = new THREE.GridHelper(2, 20, 0x888888, 0x444444);
            grid.rotateX(Math.PI / 2);
            this.scene.add(grid);

            const redMaterial = new THREE.LineBasicMaterial({
                color: 0xff0000
            });

            const greenMaterial = new THREE.LineBasicMaterial({
                color: 0x00ff00
            });

            const points = [];
            points.push(new THREE.Vector3(0, 0, 0));
            points.push(new THREE.Vector3(1, 0, 0));

            const geometry = new THREE.BufferGeometry().setFromPoints(points);

            const redLine = new THREE.Line(geometry, redMaterial);
            this.scene.add(redLine);

            const greenLine = new THREE.Line(geometry, greenMaterial);
            greenLine.rotateZ(Math.PI / 2);
            this.scene.add(greenLine);

        } else if (this.parameters.get('theme') == Themes.Simple) {
            // Plane
            const geometry = new THREE.PlaneGeometry(2, 2);
            const material = new THREE.MeshStandardMaterial({ color: 0xffffff, side: THREE.DoubleSide });

            const plane = new THREE.Mesh(geometry, material);
            plane.castShadow = false;
            plane.receiveShadow = true;

            this.scene.add(plane);
        }

        // Lights
        const light = new THREE.HemisphereLight(theme.hemisphereLightColor, 0x111122);
        this.scene.add(light);

        const pointLight = new THREE.PointLight(0xffffff, 0.3);
        pointLight.position.set(3, 3, 4);

        pointLight.castShadow = true;
        pointLight.shadow.camera.near = 0.1;
        pointLight.shadow.camera.far = 50;
        pointLight.shadow.bias = 0.0001;
        pointLight.shadow.mapSize.width = 2048;
        pointLight.shadow.mapSize.height = 2048;

        this.scene.add(pointLight);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(this.domElement.clientWidth, this.domElement.clientHeight);
        this.renderer.autoClear = false;
        this.renderer.shadowMap.enabled = this.parameters.get('shadows');
        this.renderer.shadowMap.type = THREE.PCFShadowMap;
        this.domElement.appendChild(this.renderer.domElement);

        // Label renderer
        this.labelRenderer = null;
        if (this.parameters.get('show_joint_positions')) {
            this.labelRenderer = new CSS2DRenderer();
            this.labelRenderer.setSize(this.domElement.clientWidth, this.domElement.clientHeight);
            this.labelRenderer.domElement.style.position = 'absolute';
            this.labelRenderer.domElement.style.top = '0px';
            this.domElement.appendChild(this.labelRenderer.domElement);
        }

        const renderer = (this.labelRenderer != null ? this.labelRenderer : this.renderer);

        // Scene controls
        this.cameraControl = new OrbitControls(this.camera, renderer.domElement);
        this.cameraControl.damping = 0.2;
        this.cameraControl.target = new THREE.Vector3(0, 0, 0.5);
        this.cameraControl.update();

        // Robot controls
        this.transformControls = new TransformControlsManager(this.domElement, renderer.domElement, this.camera, this.scene);
        this.transformControls.addEventListener("dragging-changed", evt => this.cameraControl.enabled = !evt.value);

        // Events handling
        new ResizeObserver(() => this._onDomElementResized()).observe(this.domElement)
        renderer.domElement.addEventListener('mousedown', evt => this._onMouseDown(evt));
        renderer.domElement.addEventListener('mouseup', evt => this._onMouseUp(evt));
        renderer.domElement.addEventListener('mousemove', evt => this._onMouseMove(evt));
        renderer.domElement.addEventListener('wheel', evt => this._onWheel(evt));
    }


    _createTcpTarget() {
        if (this.robot == null)
            return;

        this.tcpTarget = new THREE.Mesh(
            new THREE.SphereGeometry(0.2),
            new THREE.MeshBasicMaterial({
                visible: false
            })
        );

        this.tcpTarget.tag = 'tcp-target';

        this.robot.tcp.object.getWorldPosition(this.tcpTarget.position);
        this.robot.tcp.object.getWorldQuaternion(this.tcpTarget.quaternion);
        this.scene.add(this.tcpTarget);
    }


    _render() {
        requestAnimationFrame(evt => this._render());

        if (this.renderer.getPixelRatio() != window.devicePixelRatio)
            this.renderer.setPixelRatio(window.devicePixelRatio);

        if (this.stats != null)
            this.stats.update();

        TWEEN.update();

        if ((this.logmap != null) && (this.logmap.targetName != null)) {
            const mu = new THREE.Quaternion();
            const f = new THREE.Quaternion();

            const cameraOrientation = new THREE.Quaternion();
            this.camera.getWorldQuaternion(cameraOrientation);

            const target = this.targets.get(this.logmap.targetName);
            if (target != null)
                mu.copy(target.quaternion);

            if (this.robot != null)
                this.robot.tcp.object.getWorldQuaternion(f);

            this.logmap.render(this.renderer, mu, f, cameraOrientation)

            this.renderer.clearDepth();
        } else {
            this.renderer.setClearColor(ThemeParameters.get(this.parameters.get('theme')).backgroundColor, 1.0);
            this.renderer.clear();
        }

        if (this.robot != null) {
            const cameraPosition = new THREE.Vector3();
            this.camera.getWorldPosition(cameraPosition);
            this.robot.updatePositionHelpersSize(cameraPosition, this.domElement.clientWidth);
        }

        this.renderer.render(this.scene, this.camera);

        if (this.labelRenderer != null)
            this.labelRenderer.render(this.scene, this.camera);

        if ((this.renderingCallback != null) && (this.robot != null))
            this.renderingCallback(this.clock.getDelta());
    }


    _onDomElementResized() {
        const width = this.domElement.clientWidth;
        const height = this.domElement.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);

        if (this.labelRenderer != null)
            this.labelRenderer.setSize(width, height);
    }


    _onMouseDown(event) {
        if ((event.button != 0) || this.transformControls.isDragging() || !this.controlsEnabled)
            return;

        const pointer = this._getPointerPosition(event);

        this.raycaster.setFromCamera(pointer, this.camera);

        let intersects = this.raycaster.intersectObjects(this.targets.meshes, false);
        if ((intersects.length == 0) && (this.tcpTarget != null))
            intersects = this.raycaster.intersectObject(this.tcpTarget, false);

        let hoveredIntersection = null;
        if (this.interactionState == InteractionStates.JointHovering) {
            hoveredIntersection = this.raycaster.intersectObjects(this.hoveredGroup.children, false)[0];
        }

        if (intersects.length > 0) {
            const intersection = intersects[0];

            if ((hoveredIntersection == null) || (intersection.distance < hoveredIntersection.distance)) {
                if (intersection.object.tag == 'target-mesh')
                    this.transformControls.attach(intersection.object.parent);
                else
                    this.transformControls.attach(intersection.object);

                this._switchToInteractionState(InteractionStates.Manipulation);
            } else {
                if (this.jointsManipulationEnabled)
                    this._switchToInteractionState(InteractionStates.JointDisplacement);
            }
        } else {
            this.transformControls.detach();

            if (hoveredIntersection != null) {
                if (this.jointsManipulationEnabled)
                    this._switchToInteractionState(InteractionStates.JointDisplacement);
            } else {
                this._switchToInteractionState(InteractionStates.Default);
            }
        }
    }


    _onMouseUp(event) {
        if ((event.button != 0) || this.transformControls.isDragging() || !this.controlsEnabled)
            return;

        if (this.interactionState == InteractionStates.JointDisplacement) {
            const pointer = this._getPointerPosition(event);
            const hoveredGroup = this._getHoveredRobotGroup(pointer);

            if (hoveredGroup != null) {
                this._switchToInteractionState(InteractionStates.JointHovering, { group: hoveredGroup });
            } else {
                this._switchToInteractionState(InteractionStates.Default);
            }

            return;
        }

        this._switchToInteractionState(InteractionStates.Default);
    }


    _onMouseMove(event) {
        if ((this.robot == null) || this.transformControls.isDragging() || !this.controlsEnabled ||
            !this.jointsManipulationEnabled || this.transformControls.isEnabled()) {
            return;
        }

        const pointer = this._getPointerPosition(event);

        if (this.interactionState == InteractionStates.JointDisplacement) {
            const diff = new THREE.Vector2();
            diff.subVectors(pointer, this.previousPointer);

            let distance = diff.length();
            if (Math.abs(diff.x) > Math.abs(diff.y)) {
                if (diff.x < 0)
                    distance = -distance;
            } else {
                if (diff.y > 0)
                    distance = -distance;
            }

            this._changeHoveredJointPosition(2.0 * distance);
        } else {
            const hoveredGroup = this._getHoveredRobotGroup(pointer);

            if (hoveredGroup != null) {
                if (this.hoveredGroup != hoveredGroup)
                    this._switchToInteractionState(InteractionStates.JointHovering, { group: hoveredGroup });
            } else if (this.interactionState == InteractionStates.JointHovering) {
                this._switchToInteractionState(InteractionStates.Default);
            }
        }

        this.previousPointer = pointer;
    }


    _onWheel(event) {
        if (this.interactionState == InteractionStates.JointHovering) {
            this._changeHoveredJointPosition(0.2 * (event.deltaY / 106));
            event.preventDefault();
        }
    }


    _activateJointHovering(group) {
        function _highlight(object) {
            if (object.type == 'Mesh') {
                object.originalMaterial = object.material;
                object.material = object.material.clone();
                object.material.color.r *= 0xe9 / 255;
                object.material.color.g *= 0x74 / 255;
                object.material.color.b *= 0x51 / 255;
            } else if (object.type == 'Group') {
                object.children.forEach(_highlight);
            }
        }

        this.hoveredGroup = group;
        this.hoveredGroup.children.forEach(_highlight);

        this.hoveredJoint = this.robot.getJointForLink(this.hoveredGroup);
    }


    _disableJointHovering() {
        function _lessen(object) {
            if (object.type == 'Mesh') {
                object.material = object.originalMaterial;
                object.originalMaterial = undefined;
            } else if (object.type == 'Group') {
                object.children.forEach(_lessen);
            }
        }

        if (this.hoveredJoint == null)
            return;

        this.hoveredGroup.children.forEach(_lessen);
        this.hoveredGroup = null;
        this.hoveredJoint = null;
    }


    _getHoveredRobotGroup(pointer) {
        this.raycaster.setFromCamera(pointer, this.camera);

        const visuals = this.robot.arm.movable.map((x) => x.children[0].children.filter((c) => c.type == "URDFVisual")).flat();

        const groups = visuals.map((x) => x.children[0]).filter((x) => x !== undefined);

        const meshes = groups.map((x) => x.children).flat();

        let intersects = this.raycaster.intersectObjects(meshes, false);

        if (intersects.length == 0)
            return null;

        const intersection = intersects[0];
        return intersection.object.parent;
    }


    _changeHoveredJointPosition(delta) {
        const jointNames = this.robot.arm.movable.map((x) => x.name);

        let x = this.robot.getPose();
        x[jointNames.indexOf(this.hoveredJoint.name)] -= delta;
        this.robot.setPose(x);

        if (this.tcpTarget != null) {
            this.robot.tcp.object.getWorldPosition(this.tcpTarget.position);
            this.robot.tcp.object.getWorldQuaternion(this.tcpTarget.quaternion);
        }
    }


    _getPointerPosition(event) {
        const pointer = new THREE.Vector2();
        pointer.x = (event.offsetX / this.renderer.domElement.clientWidth) * 2 - 1;
        pointer.y = -(event.offsetY / this.renderer.domElement.clientHeight) * 2 + 1;

        return pointer;
    }


    _switchToInteractionState(interactionState, parameters) {
        switch (this.interactionState) {
            case InteractionStates.Default:
                break;

            case InteractionStates.Manipulation:
                break;

            case InteractionStates.JointHovering:
                if (interactionState != InteractionStates.JointDisplacement) {
                    if (this.hoveredGroup != null)
                        this._disableJointHovering();
                }
                break;

            case InteractionStates.JointDisplacement:
                if (this.hoveredGroup != null)
                    this._disableJointHovering();
                break;
        }

        this.interactionState = interactionState;

        switch (this.interactionState) {
            case InteractionStates.Default:
                this.cameraControl.enableZoom = true;
                this.cameraControl.enabled = true;
                break;

            case InteractionStates.Manipulation:
                this.cameraControl.enableZoom = true;
                this.cameraControl.enabled = true;
                break;

            case InteractionStates.JointHovering:
                this._activateJointHovering(parameters.group);
                this.cameraControl.enableZoom = false;
                this.cameraControl.enabled = true;
                break;

            case InteractionStates.JointDisplacement:
                this.cameraControl.enableZoom = true;
                this.cameraControl.enabled = false;
                break;
        }
    }
}



// Add some modules to the global scope, so they can be accessed by PyScript
globalThis.three = THREE;
globalThis.Viewer3Djs = Viewer3D;
globalThis.Themes = Themes;


// Exportations
export {Viewer3D, Themes};
