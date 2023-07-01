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
import { OutlineEffect } from 'three/examples/jsm/effects/OutlineEffect.js';
import katex from 'katex';

import { downloadFiles, downloadScene, downloadPandaRobot, loadScene } from './loading.js';
import { loadRobot } from './loader.js';
import TransformControlsManager from './control/transformcontrols.js';
import PartialIKControls from './control/partialikcontrols.js';
import FKRobot from './robots/fkrobot.js';
import Logmap from './logmap.js';
import TargetList from './helpers/targetlist.js';
import { Shapes } from './helpers/target.js';
import ObjectList from './helpers/objectlist.js';
import Arrow from './helpers/arrow.js';
import Path from './helpers/path.js';
import Point from './helpers/point.js';
import Haze from './helpers/haze.js';
import { getURL } from './utils.js';

import RobotConfiguration from './robots/configuration.js';
import PandaConfiguration from './robots/configurations/panda.js';


const InteractionStates = Object.freeze({
    Default: Symbol("default"),
    Manipulation: Symbol("manipulation"),
    JointHovering: Symbol("joint_hovering"),
    JointDisplacement: Symbol("joint_displacement"),
    LinkDisplacement: Symbol("link_displacement"),
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
        composition (list): Optional settings describing how to combine several rendering
                            layers (see below)
        ikPartialFunction (function): Function to call when a "partial ik" is needed.
                                      Signature: ikPartial(mu, nbJoints, offset)

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
                    layer: 1,
                    effect: 'outline',
                }
            ]

        Parameters for the 'outline' effect:
            thickness (float):
                thickness of the outline (default: 0.003)

            color (list of 4 floats):
                RGBA color of the outline (default: [0, 0, 0, 0])
    */
    constructor(domElement, parameters, composition, ikPartialFunction=null) {
        this.parameters = this._checkParameters(parameters);
        this.composition = this._checkComposition(composition);

        this.domElement = domElement;

        if (this.domElement == undefined)
            this.domElement = document.createElement('div');

        if (!this.domElement.classList.contains('viewer3d'))
            this.domElement.classList.add('viewer3d');

        this.camera = null;
        this.scene = [];
        this.activeLayer = 0;

        this.backgroundColor = new THREE.Color(0.0, 0.0, 0.0);
        this.skyboxScene = null;
        this.skyboxCamera = null;
        this.haze = null;

        this.physicsSimulator = null;
        this.robots = {};

        this.targets = new TargetList();
        this.arrows = new ObjectList();
        this.paths = new ObjectList();
        this.points = new ObjectList();

        this.interactionState = InteractionStates.Default;

        this.hoveredRobot = null;
        this.hoveredGroup = null;
        this.hoveredJoint = null;
        this.previousPointer = null;

        if (ikPartialFunction != null)
            this.partialIkControls = new PartialIKControls(ikPartialFunction);
        else
            this.partialIkControls = null;

        this.renderer = null;
        this.labelRenderer = null;
        this.clock = new THREE.Clock();

        this.cameraControl = null;
        this.transformControls = null;
        this.stats = null;

        this.raycaster = new THREE.Raycaster();
        this.raycaster.layers.enableAll();

        this.logmap = null;

        this.renderingCallback = null;

        this.controlsEnabled = true;
        this.endEffectorManipulationEnabled = false;
        this.jointsManipulationEnabled = true;
        this.linksManipulationEnabled = false;

        this._initScene();

        this._render();
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

        if (enabled)
            this.syncEndEffectorManipulators();
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

        if (enabled) {
            for (let name in self.robots) {
                const robot = self.robots[name];
                if (robot.tcpTarget == null)
                    robot._createTcpTarget();
            }
        }
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
            (this.interactionState == InteractionStates.JointDisplacement) ||
            (this.interactionState == InteractionStates.LinkDisplacement)) {
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


    /* Enables/disables the manipulation of the links (by click and drag).

    Note that if either 'Viewer3D.controlsEnabled' or 'Viewer3D.jointsManipulationEnabled'
    are 'false', the links can't be manipulated using the mouse regardless of the
    value of this property.

    Likewise, links manipulation is only possible when a 'partial IK function' was
    provided in the constructor.
    */
    enableLinksManipulation(enabled) {
        this.linksManipulationEnabled = enabled && (this.partialIkControls != null);

        if ((this.interactionState == InteractionStates.JointHovering) ||
            (this.interactionState == InteractionStates.JointDisplacement) ||
            (this.interactionState == InteractionStates.LinkDisplacement)) {
                this._switchToInteractionState(InteractionStates.Default);
        }
    }


    /* Indicates if the manipulation of the links is enabled (by click and drag).

    Note that if either 'Viewer3D.controlsEnabled' or 'Viewer3D.jointsManipulationEnabled'
    are 'false', the links can't be manipulated using the mouse regardless of the
    value of this property.

    Likewise, links manipulation is only possible when a 'partial IK function' was
    provided in the constructor.
    */
    isLinksManipulationEnabled() {
        return this.linksManipulationEnabled;
    }


    /* Change the layer on which new objects are created

    Each layer is drawn on top of the previous one, after clearing the depth buffer.
    The default layer (the one were the robot is) is layer 0.

    Parameters:
        layer (int): Index of the layer
    */
    activateLayer(layer) {
        while (this.composition.length < layer + 1) {
            this.composition.push(new Map([
                ['clear_depth', false],
                ['effect', null],
            ]));
        }

        this.activeLayer = layer;
    }


    loadScene(filename) {
        if (this.physicsSimulator != null) {
            const root = this.physicsSimulator.root;
            root.parent.remove(root);

            this.physicsSimulator = null;
            this.robots = {};
            this.skyboxScene = null;
            this.scene.fog = null;

            if (this.haze != null) {
                this.haze.parent.remove(this.haze);
                this.haze = null;
            }
        }

        this.physicsSimulator = loadScene(filename);

        this.scene.add(this.physicsSimulator.root);

        const stats = this.physicsSimulator.statistics;

        // Fog
        const fogCfg = this.physicsSimulator.fogSettings;
        if (fogCfg.fogEnabled) {
            this.scene.fog = new THREE.Fog(
                fogCfg.fog, fogCfg.fogStart * stats.extent, fogCfg.fogEnd * stats.extent
            );
        }

        this.clock.start();

        const paused = this.physicsSimulator.paused;

        this.physicsSimulator.paused = true;
        this.physicsSimulator.update(0);
        this.physicsSimulator.synchronize();
        this.physicsSimulator.paused = paused;

        // Update the camera position from the parameters of the scene, to see all the objects
        this.cameraControl.target.copy(stats.center);

        const camera = this.physicsSimulator.freeCameraSettings;
        const distance = 1.5 * stats.extent;

        const dir = new THREE.Vector3(
            -Math.cos(camera.azimuth * Math.PI / 180.0),
            Math.sin(-camera.elevation * Math.PI / 180.0),
            Math.sin(camera.azimuth * Math.PI / 180.0)
        ).normalize().multiplyScalar(distance);

        this.camera.position.addVectors(this.cameraControl.target, dir);
        this.camera.fov = camera.fovy;
        this.camera.near = camera.znear * stats.extent;
        this.camera.far = camera.zfar * stats.extent;
        this.camera.updateProjectionMatrix();

        this.cameraControl.update();

        // Recreate the scene in charge of rendering the background
        const textures = this.physicsSimulator.getBackgroundTextures();
        if (textures != null) {
            this.skyboxScene = new THREE.Scene();

            this.skyboxCamera = new THREE.PerspectiveCamera();
            this.skyboxCamera.fov = camera.fovy;
            this.skyboxCamera.near = this.camera.near;
            this.skyboxCamera.far = this.camera.far;
            this.skyboxCamera.updateProjectionMatrix();

            let materials = null;
            if (textures instanceof Array)
                materials = textures.map(t => new THREE.MeshBasicMaterial({ map: t, side: THREE.BackSide }));
            else
                materials = new THREE.MeshBasicMaterial({ map: textures, side: THREE.BackSide });

            const d = this.camera.far * 0.7; //(this.camera.near + this.camera.far) / 4.0;
            const geometry = new THREE.BoxGeometry(d, d, d);
            const cube = new THREE.Mesh(geometry, materials);

            this.skyboxScene.add(cube);

            if (fogCfg.hazeEnabled && (this.physicsSimulator.infinitePlane != null)) {
                this.haze = new Haze(28, fogCfg.hazeProportion, fogCfg.haze);
                this.scene.add(this.haze);
            }
        }
    }


    createRobot(name, configuration) {
        if (this.physicsSimulator == null)
            return null;

        if (name in this.robots)
            return null;

        const robot = this.physicsSimulator.createRobot(name, configuration);
        if (robot == null)
            return;

        this.physicsSimulator.simulation.forward();
        this.physicsSimulator.synchronize();

        if (this.activeLayer != 0) {
            if (!this.composition[this.activeLayer].get('cast_shadows'))
                robot.layers.disable(0);

            robot.layers.enable(this.activeLayer);
        }

        this.robots[name] = robot;

        if (this.parameters.get('show_joint_positions')) {
            let layer = this.parameters.get('joint_position_layer');
            if (layer == null)
                layer = this.activeLayer;

            robot.createJointPositionHelpers(
                this.scene, layer, this.parameters.get('joint_position_colors')
            );
        }

        if (this.endEffectorManipulationEnabled)
            robot._createTcpTarget();

        console.log(robot);

        return robot;
    }


    getRobot(name) {
        return this.robots[name];
    }


    /* Add a target to the scene, an object that can be manipulated by the user that can
    be used to define a destination position and orientation for the end-effector of the
    robot.

    Parameters:
        name (str): Name of the target
        position (Vector3): The position of the target
        orientation (Quaternion): The orientation of the target
        color (int/str): Color of the target (by default: 0x0000aa)
        shape (Shapes): Shape of the target (by default: Shapes.Cube)
    */
    addTarget(name, position, orientation, color, shape=Shapes.Cube) {
        const target = this.targets.create(name, position, orientation, color, shape);

        if (this.activeLayer != 0) {
            if (!this.composition[this.activeLayer].get('cast_shadows'))
                target.layers.disable(0);

            target.layers.enable(this.activeLayer);
        }

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
        shading (bool): Indicates if the arrow must be affected by lights (by default: false)
        headLength (Number): The length of the head of the arrow (default is 0.2 * length)
        headWidth (Number): The width of the head of the arrow (default is 0.2 * headLength)
        radius (Number): The radius of the line part of the arrow (default is 0.1 * headWidth)
    */
    addArrow(name, origin, direction, length=1, color=0xffff00, shading=false, headLength=length * 0.2,
             headWidth=headLength * 0.2, radius=headWidth*0.1
    ) {
        const arrow = new Arrow(
            name, origin, direction, length, color, shading, headLength, headWidth
        );

        if (this.activeLayer != 0) {
            if (!this.composition[this.activeLayer].get('cast_shadows'))
                arrow.layers.disable(0);

            arrow.layers.enable(this.activeLayer);
        }

        this.arrows.add(arrow);
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


    /* Add a path to the scene

    Parameters:
        name (str): Name of the path
        points (list of Vector3/list of lists of 3 numbers): Points defining the path
        radius (Number): The radius of the path (default is 0.01)
        color (int/str): Color of the path (by default: 0xffff00)
        shading (bool): Indicates if the path must be affected by lights (by default: false)
        transparent (bool): Indicates if the path must be transparent (by default: false)
        opacity (Number): Opacity level for transparent paths (between 0 and 1, default: 0.5)
    */
    addPath(name, points, radius=0.01, color=0xffff00, shading=false, transparent=false, opacity=0.5) {
        const path = new Path(name, points, radius, color, shading, transparent, opacity);

        if (this.activeLayer != 0) {
            if (!this.composition[this.activeLayer].get('cast_shadows'))
                path.layers.disable(0);

            path.layers.enable(this.activeLayer);
        }

        this.paths.add(path);
        this.scene.add(path);
        return path;
    }


    /* Remove a path from the scene.

    Parameters:
        name (str): Name of the path
    */
    removePath(name) {
        this.paths.destroy(name);
    }


    /* Returns a path from the scene.

    Parameters:
        name (str): Name of the path
    */
    getPath(name) {
        return this.paths.get(name);
    }


    /* Add a point to the scene

    Parameters:
        name (str): Name of the point
        position (Vector3): Position of the point
        radius (Number): The radius of the point (default is 0.01)
        color (int/str): Color of the point (by default: 0xffff00)
        label (str): LaTeX text to display near the point (by default: null)
        shading (bool): Indicates if the point must be affected by lights (by default: false)
        transparent (bool): Indicates if the point must be transparent (by default: false)
        opacity (Number): Opacity level for transparent points (between 0 and 1, default: 0.5)
    */
    addPoint(name, position, radius=0.01, color=0xffff00, label=null, shading=false, transparent=false, opacity=0.5) {
        const point = new Point(name, position, radius, color, label, shading, transparent, opacity);

        if (this.activeLayer != 0) {
            if (!this.composition[this.activeLayer].get('cast_shadows'))
                point.layers.disable(0);

            point.layers.enable(this.activeLayer);
        }

        this.points.add(point);
        this.scene.add(point);
        return point;
    }


    /* Remove a point from the scene.

    Parameters:
        name (str): Name of the point
    */
    removePoint(name) {
        this.points.destroy(name);
    }


    /* Returns a point from the scene.

    Parameters:
        name (str): Name of the point
    */
    getPoint(name) {
        return this.points.get(name);
    }


    enableLogmap(robot, target, position='left', size=null) {
        if (typeof(robot) == "string")
            robot = this.getRobot(robot);

        if (typeof(target) == "string")
            target = this.getTarget(target);

        this.logmap = new Logmap(this.domElement, robot, target, size, position);
    }


    disableLogmap() {
        this.logmap = null;
    }


    syncEndEffectorManipulators() {
        const _tmpVector3 = new THREE.Vector3();
        const _tmpQuaternion = new THREE.Quaternion();
        const _tmpQuaternion2 = new THREE.Quaternion();

        for (let name in self.robots) {
            const robot = self.robots[name];
            if (robot.tcpTarget != null) {
                robot.tcp.getWorldPosition(_tmpVector3);
                robot.tcp.getWorldQuaternion(_tmpQuaternion);

                robot.arm.visual.links[0].worldToLocal(_tmpVector3);
                robot.tcpTarget.position.copy(_tmpVector3);

                robot.arm.visual.links[0].getWorldQuaternion(_tmpQuaternion2);
                robot.tcpTarget.quaternion.multiplyQuaternions(_tmpQuaternion2.invert(), _tmpQuaternion);
            }
        }
    }


    _checkParameters(parameters) {
        if (parameters == null)
            parameters = new Map();
        else if (!(parameters instanceof Map))
            parameters = new Map(Object.entries(parameters));

        const defaults = new Map([
            ['joint_position_colors', []],
            ['joint_position_layer', null],
            ['robot_use_toon_shader', false],
            ['robot_use_light_toon_shader', false],
            ['shadows', true],
            ['show_joint_positions', false],
            ['statistics', false],
        ]);

        return new Map([...defaults, ...parameters]);
    }


    _checkComposition(composition) {
        if (composition == null)
            composition = [];

        const defaults = new Map([
            ['cast_shadows', true],
            ['clear_depth', false],
            ['effect', null],
            ['effect_parameters', null],
        ]);

        const result = [];

        // Apply defaults
        for (let i = 0; i < composition.length; ++i) {
            let entry = composition[i];
            if (!(entry instanceof Map))
                entry = new Map(Object.entries(entry));

            const layer = entry.get('layer');
            result[layer] = new Map([...defaults, ...entry]);
        }

        // Ensure that all known layers have parameters
        for (let i = 0; i < result.length; ++i) {
            if (result[i] == null)
                result[i] = new Map(defaults);
        }

        if (result.length == 0)
            result.push(new Map(defaults));

        // Ensure that the first layer clears the depth buffer
        result[0].set('clear_depth', true);

        return result;
    }


    _initScene() {
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
        this.camera.position.set(1, 1, -2);

        this.scene = new THREE.Scene();

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(this.domElement.clientWidth, this.domElement.clientHeight);
        this.renderer.autoClear = false;
        this.renderer.shadowMap.enabled = this.parameters.get('shadows');
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.domElement.appendChild(this.renderer.domElement);

        // Label renderer
        this.labelRenderer = new CSS2DRenderer();
        this.labelRenderer.setSize(this.domElement.clientWidth, this.domElement.clientHeight);
        this.labelRenderer.domElement.style.position = 'absolute';
        this.labelRenderer.domElement.style.top = '0px';
        this.domElement.appendChild(this.labelRenderer.domElement);

        // Effects
        this.effects = [];

        for (let i = 0; i < this.composition.length; ++i) {
            const effect = this.composition[i].get('effect');

            let effect_parameters = this.composition[i].get('effect_parameters');
            if (effect_parameters == null)
                effect_parameters = new Map();
            else if (!(effect_parameters instanceof Map))
                effect_parameters = new Map(Object.entries(entry));

            if (effect == 'outline') {
                const color = effect_parameters.get('color') || [0.0, 0.0, 0.0, 0.0];
                const thickness = effect_parameters.get('thickness') || 0.003;

                this.effects.push(
                    new OutlineEffect(
                        this.renderer,
                        {
                            defaultAlpha: color[3],
                            defaultThickness: thickness,
                            defaultColor: color.slice(0, 3)
                        }
                    )
                );
            } else {
                this.effects.push(null);
            }
        }

        // Scene controls
        const renderer = this.labelRenderer;

        this.cameraControl = new OrbitControls(this.camera, renderer.domElement);
        this.cameraControl.damping = 0.2;
        this.cameraControl.maxPolarAngle = Math.PI / 2.0 + 0.2;
        this.cameraControl.target = new THREE.Vector3(0, 0.5, 0);
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

        document.addEventListener("visibilitychange", () => {
            if (document.hidden) {
                this.clock.stop();
                this.clock.autoStart = true;
            }
        });

        this.activateLayer(0);
    }


    _render() {
        // Retrieve the time elapsed since the last frame
        const startTime = this.clock.startTime;
        const oldTime = this.clock.oldTime;
        const mustAdjustClock = !this.clock.running;

        const delta = this.clock.getDelta();

        if (mustAdjustClock) {
            this.clock.startTime = startTime + this.clock.oldTime - oldTime;

            this.clock.elapsedTime = (this.clock.oldTime - this.clock.startTime) * 0.001;

            if (this.physicsSimulator != null)
                this.physicsSimulator.time = this.clock.elapsedTime;
        }

        // Ensure that the pixel ratio is still correct (might change when the window is
        // moved from one screen to another)
        if (this.renderer.getPixelRatio() != window.devicePixelRatio)
            this.renderer.setPixelRatio(window.devicePixelRatio);

        // Update the statistics (if necessary)
        if (this.stats != null)
            this.stats.update();

        // Update the tween variables
        TWEEN.update();

        // Update the physics simulator
        if (this.physicsSimulator != null) {
            this.physicsSimulator.update(this.clock.elapsedTime);
            this.physicsSimulator.synchronize();
        }

        // Ensure that the camera isn't below the floor
        this.cameraControl.target.y = Math.max(this.cameraControl.target.y, 0.0);

        // Update the robot joints visualisation (if necessary)
        const cameraPosition = new THREE.Vector3();
        this.camera.getWorldPosition(cameraPosition);

        for (const name in this.robots)
            this.robots[name].synchronize(cameraPosition, this.domElement.clientWidth);

        if (this.physicsSimulator != null) {
            // Update the headlight position (if necessary)
            if (this.physicsSimulator.headlight != null) {
                this.physicsSimulator.headlight.position.copy(this.camera.position);
                this.physicsSimulator.headlight.target.position.copy(this.cameraControl.target);
            }

            // Render the background
            const fogCfg = this.physicsSimulator.fogSettings;

            if (fogCfg.fogEnabled) {
                this.renderer.setClearColor(fogCfg.fog, 1.0);
                this.renderer.clear();
            }
            else if (this.skyboxScene) {
                this.renderer.clear();

                const position = new THREE.Vector3();

                this.camera.getWorldPosition(position);
                this.skyboxCamera.position.y = position.y;

                this.camera.getWorldQuaternion(this.skyboxCamera.quaternion);

                this.renderer.render(this.skyboxScene, this.skyboxCamera);

                if (this.haze != null) {
                    const skyboxDistance = this.skyboxCamera.far * 0.7;

                    const position2 = new THREE.Vector3();
                    this.physicsSimulator.infinitePlane.getWorldPosition(position2);

                    position.sub(position2);

                    const elevation = this.physicsSimulator.infinitePlane.scale.y * position.y / skyboxDistance;

                    this.haze.scale.set(skyboxDistance, elevation, skyboxDistance);
                }
            } else {
                this.renderer.setClearColor(this.backgroundColor, 1.0);
                this.renderer.clear();
            }
        } else {
            this.renderer.setClearColor(this.backgroundColor, 1.0);
            this.renderer.clear();
        }

        // Render the scenes
        this.camera.layers.disableAll();
        const disabledMaterials = [];

        for (let i = 0; i < this.composition.length; ++i) {
            const layerConfig = this.composition[i];

            if (layerConfig.get('clear_depth'))
                this.renderer.clearDepth();

            this.camera.layers.enable(i);

            if (i == 0) {
                const objects = [
                    Object.keys(this.robots).map(name => { return this.robots[name]; }),
                    Object.keys(this.targets.targets).map(name => { return this.targets.get(name); }),
                    Object.keys(this.arrows.objects).map(name => { return this.arrows.get(name); }),
                    Object.keys(this.paths.objects).map(name => { return this.paths.get(name); }),
                    Object.keys(this.points.objects).map(name => { return this.points.get(name); }),
                ].flat();

                for (let obj of objects) {
                    if ((obj.layers.mask > 1) && (obj.layers.mask & 0x1 == 1))
                        obj._disableVisibility(disabledMaterials);
                }
            }

            const effect = this.effects[i];
            if (effect != null)
                effect.render(this.scene, this.camera);
            else
                this.renderer.render(this.scene, this.camera);

            if (i == 0) {
                for (let material of disabledMaterials) {
                    material.colorWrite = true;
                    material.depthWrite = true;
                }
            }

            this.camera.layers.disable(i);
        }

        // Display the labels
        this.camera.layers.enable(31);
        this.labelRenderer.render(this.scene, this.camera);
        this.camera.layers.disable(31);

        // Update the logmap visualisation (if necessary)
        if (this.logmap) {
            const cameraOrientation = new THREE.Quaternion();
            this.camera.getWorldQuaternion(cameraOrientation);

            this.renderer.clearDepth();
            this.logmap.render(this.renderer, cameraOrientation)
        }

        // Notify the listener (if necessary)
        if (this.renderingCallback != null)
            this.renderingCallback(delta, this.clock.elapsedTime);

        // Request another animation frame
        requestAnimationFrame(() => this._render());
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
        if (intersects.length == 0) {
            const tcpTargets = [];
            for (let name in this.robots) {
                const robot = this.robots[name];
                if (robot.tcpTarget != null)
                    tcpTargets.push(robot.tcpTarget);
            }

            if (tcpTargets.length > 0)
                intersects = this.raycaster.intersectObjects(tcpTargets, false);
        }

        let hoveredIntersection = null;
        if (this.interactionState == InteractionStates.JointHovering) {
            hoveredIntersection = this.raycaster.intersectObjects(this.hoveredGroup.children, false)[0];
        }

        let intersection = null;

        if (intersects.length > 0) {
            intersection = intersects[0];

            if ((hoveredIntersection == null) || (intersection.distance < hoveredIntersection.distance)) {
                if (intersection.object.tag == 'target-mesh')
                    this.transformControls.attach(intersection.object.parent);
                else
                    this.transformControls.attach(intersection.object);

                this._switchToInteractionState(InteractionStates.Manipulation);
            } else {
                intersection = null;
            }
        }

        if (intersection == null) {
            this.transformControls.detach();

            if (hoveredIntersection != null) {
                if (this.jointsManipulationEnabled) {
                    if (this.linksManipulationEnabled) {
                        this._switchToInteractionState(InteractionStates.LinkDisplacement);

                        const jointIndex = this.hoveredRobot.arm.joints.indexOf(this.hoveredJoint);
                        const joint = this.hoveredRobot.arm.visual.joints[jointIndex];

                        const direction = new THREE.Vector3();
                        this.camera.getWorldDirection(direction);

                        this.partialIkControls.setup(
                            this.hoveredRobot,
                            joint.worldToLocal(hoveredIntersection.point.clone()),
                            jointIndex + 1,
                            hoveredIntersection.point,
                            direction
                        );
                    } else {
                        this._switchToInteractionState(InteractionStates.JointDisplacement);
                    }
                }
            } else {
                this._switchToInteractionState(InteractionStates.Default);
            }
        }
        
        event.preventDefault();
    }


    _onMouseUp(event) {
        if ((event.button != 0) || this.transformControls.isDragging() || !this.controlsEnabled)
            return;

        if ((this.interactionState == InteractionStates.JointDisplacement) ||
            (this.interactionState == InteractionStates.LinkDisplacement)) {
            const pointer = this._getPointerPosition(event);
            const [hoveredRobot, hoveredGroup] = this._getHoveredRobotGroup(pointer);

            if (hoveredGroup != null) {
                this._switchToInteractionState(InteractionStates.JointHovering, { group: hoveredGroup });
            } else {
                this._switchToInteractionState(InteractionStates.Default);
            }

            return;
        }

        this._switchToInteractionState(InteractionStates.Default);

        event.preventDefault();
    }


    _onMouseMove(event) {
        if (this.transformControls.isDragging() || !this.controlsEnabled ||
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

        } else if (this.interactionState == InteractionStates.LinkDisplacement) {
            this.raycaster.setFromCamera(pointer, this.camera);
            this.partialIkControls.process(this.raycaster);

        } else {
            const [hoveredRobot, hoveredGroup] = this._getHoveredRobotGroup(pointer);

            if (hoveredGroup != null) {
                if (this.hoveredGroup != hoveredGroup)
                    this._switchToInteractionState(InteractionStates.JointHovering, { robot: hoveredRobot, group: hoveredGroup });
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


    _activateJointHovering(robot, group) {
        function _highlight(object) {
            if (object.type == 'Mesh') {
                object.originalMaterial = object.material;
                object.material = object.material.clone();
                object.material.color.r *= 0xe9 / 255;
                object.material.color.g *= 0x74 / 255;
                object.material.color.b *= 0x51 / 255;
            }
        }

        this.hoveredRobot = robot;
        this.hoveredGroup = group;
        this.hoveredGroup.children.forEach(_highlight);

        this.hoveredJoint = this.hoveredGroup.jointId;
    }


    _disableJointHovering() {
        function _lessen(object) {
            if (object.type == 'Mesh') {
                object.material = object.originalMaterial;
                object.originalMaterial = undefined;
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

        const meshes = Object.values(this.robots).map((r) => r.arm.visual.meshes).flat()
                                                 .filter((mesh) => mesh.parent.jointId !== undefined);

        let intersects = this.raycaster.intersectObjects(meshes, false);
        if (intersects.length == 0)
            return [null, null];

        const intersection = intersects[0];

        const group = intersection.object.parent;
        const robot = Object.values(this.robots).filter((r) => r.arm.links.indexOf(group.bodyId) >= 0)[0];

        return [robot, group];
    }


    _changeHoveredJointPosition(delta) {
        let ctrl = this.hoveredRobot.getControl();
        ctrl[this.hoveredRobot.arm.joints.indexOf(this.hoveredJoint)] -= delta;
        this.hoveredRobot.setControl(ctrl);

        this.syncEndEffectorManipulators();
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
                if ((interactionState != InteractionStates.JointDisplacement) &&
                    (interactionState != InteractionStates.LinkDisplacement)) {
                    if (this.hoveredGroup != null)
                        this._disableJointHovering();
                }
                break;

            case InteractionStates.JointDisplacement:
            case InteractionStates.LinkDisplacement:
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
                this._activateJointHovering(parameters.robot, parameters.group);
                this.cameraControl.enableZoom = false;
                this.cameraControl.enabled = true;
                break;

            case InteractionStates.JointDisplacement:
            case InteractionStates.LinkDisplacement:
                this.cameraControl.enableZoom = true;
                this.cameraControl.enabled = false;
                break;
        }
    }
}


function initPyScript() {
    // Add some modules to the global scope, so they can be accessed by PyScript
    globalThis.three = THREE;
    globalThis.katex = katex;
    globalThis.Viewer3Djs = Viewer3D;
    globalThis.Shapes = Shapes;

    globalThis.configs = {
        RobotConfiguration: RobotConfiguration,
        Panda: PandaConfiguration,
    };

    // Process the importmap to avoid errors from PyScript
    const scripts = document.getElementsByTagName('script');
    for (let script of scripts) {
        if (script.type == 'importmap') {
            const importmap = JSON.parse(script.innerText);
            delete importmap['imports']['three/examples/jsm/'];
            delete importmap['imports']['mujoco'];
            script.innerText = JSON.stringify(importmap);
            break;
        }
    }

    // Add the PyScript script to the document
    const script = document.createElement('script');
    script.src = 'https://pyscript.net/latest/pyscript.min.js';
    script.type = 'text/javascript';
    document.body.appendChild(script);
}


// Add some needed CSS files to the HTML page
const cssFiles = [
    getURL('css/style.css'),
    'https://cdn.jsdelivr.net/npm/katex@0.16.2/dist/katex.min.css'
];

cssFiles.forEach(css => {
    const link = document.createElement('link');
    link.rel = 'stylesheet';
    link.type = 'text/css';
    link.href = css;
    document.getElementsByTagName('HEAD')[0].appendChild(link);
});



// Exportations
export { Viewer3D, Shapes, initPyScript, downloadFiles, downloadScene, downloadPandaRobot };
