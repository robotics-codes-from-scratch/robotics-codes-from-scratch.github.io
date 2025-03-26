import * as THREE from 'three';
import { Vector2, Color, WebGLRenderTarget, MeshDepthMaterial, DoubleSide, RGBADepthPacking, NoBlending, HalfFloatType, UniformsUtils, ShaderMaterial, Matrix4, Vector3, NormalBlending } from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import Stats from 'three/examples/jsm/libs/stats.module.js';
import * as TWEEN from 'three/examples/jsm/libs/tween.module.js';
import { CSS2DObject, CSS2DRenderer } from 'three/examples/jsm/renderers/CSS2DRenderer.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import katex from 'katex';
import load_mujoco from 'mujoco';
import * as math from 'mathjs';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';
import { Pass, FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';
import { CopyShader } from 'three/examples/jsm/shaders/CopyShader.js';

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



// Load the MuJoCo Module
const mujoco = await load_mujoco();

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



function readFile(filename, binary=false) {
    try {
        const stat = mujoco.FS.stat(filename);
    } catch (ex) {
        return null;
    }

    const content = mujoco.FS.readFile(filename);

    if (!binary)
    {
        const textDecoder = new TextDecoder("utf-8");
        return textDecoder.decode(content);
    }

    return content;
}


function writeFile(filename, content) {
    mujoco.FS.writeFile(filename, content);
}


function mkdir(path) {
    const parts = path.split("/");

    let current = parts[0];
    if (path[0] == "/")
        current = "/" + current;

    let i = 0;

    while (i < parts.length) {
        try {
            const stat = mujoco.FS.stat(current);
        } catch (ex) {
            mujoco.FS.mkdir(current);
        }

        i++;
        if (i < parts.length)
            current += "/" + parts[i];
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */




/* Arcosine redefinition to make sure the distance between antipodal quaternions is zero
*/
function acoslog(x) {
    let y = math.acos(Math.min(x, 1.0));

    if (math.typeOf(y) == 'Complex')
        return NaN;

    if (x < 0.0)
        y = y - math.pi;

    return y;
}



/* Logarithmic map for S^3 manifold (with e in tangent space)
*/
function logmap_S3(x, x0) {

    function _dQuatToDxJac(q) {
        // Jacobian from quaternion velocities to angular velocities.
        // q is wxyz!
        return math.matrix([
            [-q.get([1]), q.get([0]), -q.get([3]), q.get([2])],
            [-q.get([2]), q.get([3]), q.get([0]), -q.get([1])],
            [-q.get([3]), -q.get([2]), q.get([1]), q.get([0])],
        ]);
    }

    if (math.typeOf(x) == 'DenseMatrix')
        x = math.reshape(x, [4]).toArray();

    if (math.typeOf(x0) == 'DenseMatrix')
        x0 = math.reshape(x0, [4]).toArray();

    // Code below is for quat as wxyz so need to transform it!
    x = math.matrix([x[3], x[0], x[1], x[2]]);
    x0 = math.matrix([x0[3], x0[0], x0[1], x0[2]]);

    const x0Tx = math.multiply(math.transpose(x0), x);

    const th = acoslog(x0Tx);

    let u = math.subtract(x, math.multiply(x0Tx, x0));

    // Avoid numerical issue with small numbers
    if (math.norm(u) < 1e-7)
        return math.zeros(3);

    u = math.divide(math.multiply(th, u), math.norm(u));

    const H = _dQuatToDxJac(x0);
    return math.squeeze(math.multiply(math.multiply(2, math.squeeze(H)), u))
}



/* Logarithmic map for R^3 x S^3 manifold (with e in tangent space)
*/
function logmap$1(f, f0) {
    let e;

    if (f.size().length == 1) {
        const indices1 = math.index(math.range(0, 3));
        const indices2 = math.index(math.range(3, 6));
        const indices3 = math.index(math.range(3, f.size()[0]));

        e = math.zeros(6);
        e.subset(indices1, math.subtract(f.subset(indices1), f0.subset(indices1)));
        e.subset(indices2, logmap_S3(f.subset(indices3), f0.subset(indices3)));

    } else {
        const N = f.size()[1];
        const M = f.size()[0];

        const indices1 = math.index(math.range(0, 3), math.range(0, N));
        math.index(math.range(3, 6));
        math.index(math.range(3, f.size()[0]));

        e = math.zeros(6, f.size()[1]);

        e.subset(indices1, math.subtract(f.subset(indices1), f0.subset(indices1)));

        for (let t = 0; t < N; ++t)
            e.subset(math.index(math.range(3, 6), t), logmap_S3(f.subset(math.index(math.range(3, M), t)), f0.subset(math.index(math.range(3, M), t))));
    }

    return e;
}

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



const _tmpVector3$2 = new THREE.Vector3();
const _tmpQuaternion$2 = new THREE.Quaternion();



class KinematicChain {

    constructor(robot, joint=null, tool=null) {
        this.robot = robot;
        this.tool = tool;

        let segment = null;
        let index = null;

        if (tool != null) {
            segment = robot.tools[tool].parent;
            index = segment.joints.length - 1;
        } else {
            segment = robot._getSegmentOfJoint(joint);
            index = segment.joints.indexOf(joint);
        }

        this.joints = segment.joints.slice(0, index+1);
        this.actuators = segment.actuators.slice(0, index+1);
        this.limits = segment.limits.slice(0, index+1);

        while (segment.parent != -1) {
            segment = robot.segments[segment.parent];
            if (segment.joints.length > 0) {
                this.joints = segment.joints.concat(this.joints);
                this.actuators = segment.actuators.concat(this.actuators);
                this.limits = segment.limits.concat(this.limits);
            }
        }
    }


    /* Performs Forward Kinematics, on a subset of the joints

    Parameters:
        positions (array): The joint positions
        offset (Vector3): Optional, an offset from the last joint

    Returns:
        A tuple of a Vector3 and a Quaternion: (position, orientation)
    */
    fkin(positions, offset=null) {
        if (math.typeOf(positions) == 'DenseMatrix')
            positions = positions.toArray();
        else if (math.typeOf(positions) == 'number')
            positions = [positions];

        // Check the input
        if (positions.length > this.joints.length)
            throw new Error('The number of joint positions must be less or equal than the number of movable joints');

        // Set the joint positions
        const nbJoints = positions.length;

        for (let i = 0; i < nbJoints; ++i)
            this.robot._setJointPosition(this.joints[i], positions[i]);

        // Retrieve the position and orientation of the last joint
        if ((positions.length == this.joints.length) && (offset == null)) {
            this.robot.fk.tcps[this.tool].getWorldPosition(_tmpVector3$2);
            this.robot.fk.tcps[this.tool].getWorldQuaternion(_tmpQuaternion$2);
        } else {
            const bodyId = this.robot.fk.joints.filter((j) => j.jointId == this.joints[positions.length - 1])[0].children[0].bodyId;
            const link = this.robot.fk.links.filter((l) => l.bodyId == bodyId)[0];

            link.getWorldPosition(_tmpVector3$2);
            link.getWorldQuaternion(_tmpQuaternion$2);

            if (offset != null) {
                let offset2;

                if (math.typeOf(offset) == 'DenseMatrix')
                    offset2 = new THREE.Vector3().fromArray(offset.toArray());
                else if (Array.isArray(offset))
                    offset2 = new THREE.Vector3().fromArray(offset);
                else
                    offset2 = offset.clone();

                offset2.applyQuaternion(_tmpQuaternion$2);
                _tmpVector3$2.add(offset2);
            }
        }

        return [
            _tmpVector3$2.x, _tmpVector3$2.y, _tmpVector3$2.z,
            _tmpQuaternion$2.x, _tmpQuaternion$2.y, _tmpQuaternion$2.z, _tmpQuaternion$2.w,
        ];
    }


    ik(mu, nbJoints=null, offset=null, limit=5, dt=0.01, successDistance=1e-4, damping=false) {
        let x = math.matrix(Array.from(this.getControl()));
        const startx = math.matrix(x);

        if (math.typeOf(mu) == 'Array')
            mu = math.matrix(mu);

        if (nbJoints == null)
            nbJoints = x.size()[0];

        const jointIndices = math.index(math.range(0, nbJoints));

        let indices = math.index(math.range(0, 7));
        let jacobianIndices = math.index(math.range(0, 6), math.range(0, nbJoints));
        if (mu.size()[0] == 3) {
            indices = math.index(math.range(0, 3));
            jacobianIndices = math.index(math.range(0, 3), math.range(0, nbJoints));
        } else if (mu.size()[0] == 4) {
            indices = math.index(math.range(3, 7));
            jacobianIndices = math.index(math.range(3, 6), math.range(0, nbJoints));
        }

        damping = damping || (this.tool == null) || (nbJoints < x.size()[0]);

        let done = false;
        let i = 0;
        let diff;
        let pinvJ;
        let u;
        while (!done && ((limit == null) || (i < limit))) {
            const f = math.matrix(this.fkin(x.subset(jointIndices), offset));

            if (mu.size()[0] == 3)
                diff = math.subtract(mu, f.subset(indices));
            else if (mu.size()[0] == 4)
                diff = logmap_S3(mu, f.subset(indices));
            else
                diff = logmap$1(mu, f);

            let J = this.Jkin(x.subset(jointIndices), offset);
            J = J.subset(jacobianIndices);

            if (!damping) {
                try {
                    pinvJ = math.pinv(J);
                }
                catch(err) {
                    // Sometimes mathjs fail to compute the pseudoinverse, so as a fallback we'll compute it
                    // manually using a damping factor
                    damping = true;
                }
            }

            if (damping) {
                // Damped pseudoinverse
                const JT = math.transpose(J);

                pinvJ = math.multiply(
                    math.inv(
                        math.add(
                            math.multiply(JT, J),
                            math.multiply(math.identity(nbJoints), 1e-2)
                        )
                    ),
                    JT
                );
            }

            u = math.multiply(math.multiply(pinvJ, diff), 0.1 / dt);  // Velocity command, with a 0.1 gain to not overshoot the target

            x.subset(jointIndices, math.add(x.subset(jointIndices), math.multiply(u, dt)).subset(jointIndices));

            i++;

            if (math.norm(math.subtract(x, startx)) < successDistance)
                done = true;
        }

        startx.subset(jointIndices, x.subset(jointIndices));
        this.setControl(startx.toArray());

        return done;
    }


    /* Jacobian with numerical computation, on a subset of the joints
    */
    Jkin(positions, offset=null) {
        const eps = 1e-6;

        if (math.typeOf(positions) == 'Array')
            positions = math.matrix(positions);
        else if (math.typeOf(positions) == 'number')
            positions = math.matrix([ positions ]);

        const D = positions.size()[0];

        positions = math.reshape(positions, [D, 1]);

        // Matrix computation
        const F1 = math.zeros(7, D);
        const F2 = math.zeros(7, D);

        const f0 = math.matrix(this.fkin(positions, offset));

        for (let i = 0; i < D; ++i) {
            F1.subset(math.index(math.range(0, 7), i), f0);

            const diff = math.zeros(D);
            diff.set([i, 0], eps);

            F2.subset(math.index(math.range(0, 7), i), this.fkin(math.add(positions, diff), offset));
        }

        let J = math.divide(logmap$1(F2, F1), eps);

        if (J.size().length == 1)
            J = math.reshape(J, [J.size()[0], 1]);

        return J;
    }


    getJointPositions() {
        return this.robot._simulator.getJointPositions(this.joints);
    }


    getControl() {
        return this.robot._simulator.getControl(this.actuators);
    }


    setControl(control) {
        const ctrl = control.map(
            (v, i) => (Math.abs(this.limits[i][0]) > 1e-6) || (Math.abs(this.limits[i][1]) > 1e-6) ?
                            Math.min(Math.max(v, this.limits[i][0]), this.limits[i][1]) :
                            v
        );

        const nbJoints = control.length;

        if (this.robot._simulator.paused)
            this.robot._simulator.setJointPositions(ctrl, this.joints.slice(0, nbJoints));

        this.robot._simulator.setControl(ctrl, this.actuators.slice(0, nbJoints));
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



const axisZ = new THREE.Vector3(0, 0, 1);


class JointPositionGeometry extends THREE.BufferGeometry {

    constructor(radius = 1, segments = 32, thetaStart = 0, thetaLength = Math.PI * 2) {

        super();

        this.type = 'JointPositionGeometry';

        this.parameters = {
            radius: radius,
            segments: segments,
            thetaStart: thetaStart,
            thetaLength: thetaLength
        };

        this.segments = Math.max(3, segments);

        // buffers
        const indices = [];
        const vertices = [];
        const normals = [];

        // center point
        vertices.push(0, 0, 0);
        normals.push(0, 1, 0);

        for (let s = 0, i = 3; s <= segments; s++, i += 3) {
            // vertex
            vertices.push(0, 0, 0);

            // normal
            normals.push(0, 1, 0);
        }

        // indices
        for (let i = 1; i <= segments; i++)
            indices.push(i, i + 1, 0);

        // position buffer
        this.positionBuffer = new THREE.Float32BufferAttribute(vertices, 3);
        this.positionBuffer.setUsage(THREE.DynamicDrawUsage);

        this.update(thetaStart, thetaLength);

        // build geometry
        this.setIndex(indices);
        this.setAttribute('position', this.positionBuffer);
        this.setAttribute('normal', new THREE.Float32BufferAttribute( normals, 3 ) );
    }

    update(thetaStart = 0, thetaLength = Math.PI * 2) {
        for (let s = 0, i = 3; s <= this.segments; s++, i += 3) {
            const segment = thetaStart + s / this.segments * thetaLength;

            this.positionBuffer.array[i] = this.parameters.radius * Math.cos(segment);
            this.positionBuffer.array[i+2] =  -this.parameters.radius * Math.sin(segment);
        }

        this.positionBuffer.needsUpdate = true;
    }

}



class JointPositionHelper extends THREE.Object3D {

    constructor(scene, layer, jointId, jointIndex, axis, jointPosition, invert=false, color=0xff0000, offset=0.0) {
        super();

        this.layers.disableAll();
        this.layers.enable(layer);

        this.isJointPositionHelper = true;
        this.type = 'JointPositionHelper';

        this.jointId = jointId;
        this.invert = invert || false;
        this.previousDistanceToCamera = null;
        this.previousPosition = null;

        this.origin = new THREE.Object3D();
        this.origin.translateZ(offset);
        this.add(this.origin);

        const zAxis = new THREE.Vector3(0, 0, 1);
        this.origin.quaternion.setFromUnitVectors(zAxis, axis);

        if (typeof color === 'string') {
            if (color[0] == '#')
                color = color.substring(1);
            color = Number('0x' + color);
        }

        color = new THREE.Color(color);

        const lineMaterial = new THREE.LineBasicMaterial({
            color: color
        });

        const points = [];
        points.push(new THREE.Vector3(0, 0, 0));
        points.push(new THREE.Vector3(0.2, 0, 0));

        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);


        const circleMaterial = new THREE.MeshBasicMaterial({
            color: color,
            opacity: 0.4,
            transparent: true,
            side: THREE.DoubleSide
        });


        this.startLine = new THREE.Line(lineGeometry, lineMaterial);
        this.startLine.layers = this.layers;
        this.origin.add(this.startLine);

        this.endLine = new THREE.Line(lineGeometry, lineMaterial);
        this.endLine.layers = this.layers;
        this.origin.add(this.endLine);

        if (invert)
            this.endLine.setRotationFromAxisAngle(axisZ, Math.PI);

        const circleGeometry = new JointPositionGeometry(0.2, 16, -jointPosition, jointPosition);

        this.circle = new THREE.Mesh(circleGeometry, circleMaterial);
        this.circle.layers = this.layers;
        this.circle.rotateX(Math.PI / 2);
        this.origin.add(this.circle);


        this.labelRotator = new THREE.Object3D();
        this.origin.add(this.labelRotator);

        this.labelElement = document.createElement('div');
        this.labelElement.style.fontSize = '1vw';

        katex.render(String.raw`\color{#` + color.getHexString() + `}x_{` + jointIndex + `}`, this.labelElement, {
            throwOnError: false
        });

        this.label = new CSS2DObject(this.labelElement);
        this.label.position.set(0.24, 0, 0);
        this.labelRotator.add(this.label);

        this.label.layers.disableAll();
        this.label.layers.enable(31);

        scene.add(this);
    }


    destroy() {
        this.parent.remove(this);
        this.labelElement.remove();
    }


    updateTransforms(joint) {
        joint.getWorldPosition(this.position);
        joint.getWorldQuaternion(this.quaternion);
    }


    updateJointPosition(jointPosition) {
        if ((this.previousPosition != null) && (Math.abs(jointPosition - this.previousPosition) < 1e-6))
            return;

        if (this.invert) {
            this.startLine.setRotationFromAxisAngle(axisZ, Math.PI - jointPosition);
            this.circle.geometry.update(Math.PI - jointPosition, jointPosition);
            this.labelRotator.setRotationFromAxisAngle(axisZ, Math.PI - jointPosition / 2);
        } else {
            this.startLine.setRotationFromAxisAngle(axisZ, -jointPosition);
            this.circle.geometry.update(-jointPosition, jointPosition);
            this.labelRotator.setRotationFromAxisAngle(axisZ, -jointPosition / 2);
        }

        this.previousPosition = jointPosition;
    }


    updateSize(cameraPosition, elementWidth) {
        const position = new THREE.Vector3();
        this.getWorldPosition(position);

        const dist = cameraPosition.distanceToSquared(position);

        const maxDist = 0.21 + 0.03 * 1000 / elementWidth;

        if (dist > 30.0) {
            if (this.previousDistanceToCamera != 30) {
                this.labelElement.style.fontSize = '0.7vw';
                this.label.position.x = maxDist;
                this.previousDistanceToCamera = 30;
            }
        } else if (dist > 10.0) {
            if (this.previousDistanceToCamera != 10) {
                this.labelElement.style.fontSize = '0.8vw';
                this.label.position.x = maxDist;
                this.previousDistanceToCamera = 10;
            }
        } else if (dist > 5.0) {
            if (this.previousDistanceToCamera != 5) {
                this.labelElement.style.fontSize = '0.9vw';
                this.label.position.x = maxDist;
                this.previousDistanceToCamera = 5;
            }
        } else {
            if (this.previousDistanceToCamera != 0) {
                this.labelElement.style.fontSize = '1vw';
                this.previousDistanceToCamera = 0;
            }

            this.label.position.x = 0.21 + 0.03 * 1000 / elementWidth * Math.max(dist, 0.001) / 5.0;
        }
    }


    _disableVisibility(materials) {
        this.startLine.material.colorWrite = false;
        this.startLine.material.depthWrite = false;

        this.circle.material.colorWrite = false;
        this.circle.material.depthWrite = false;

        materials.push(this.startLine.material, this.circle.material);
    }

}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



function getURL(path) {
    let url = new URL(import.meta.url);
    return url.href.substring(0, url.href.lastIndexOf('/')) + '/' + path;
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 * SPDX-FileCopyrightText: Copyright © 2022 Nikolas Dahn
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 * SPDX-FileContributor: Nikolas Dahn
 *
 * SPDX-License-Identifier: MIT
 *
 * This file is a modification of the one implemented in https://github.com/ndahn/Rocksi
 *
 */



const _tmpVector3$1 = new THREE.Vector3();
const _tmpQuaternion$1 = new THREE.Quaternion();
const _tmpQuaternion2 = new THREE.Quaternion();



class Robot {

    constructor(name, configuration, physicsSimulator) {
        this.name = name;
        this.configuration = configuration;
        this.controlsEnabled = true;

        this.segments = [];

        this.joints = [];
        this.actuators = [];
        this.links = [];

        this.names = {
            joints: [],
            actuators: [],
            links: [],
        },

        this.tools = [];
        this.toolsEnabled = false;
        this.toolbar = null;

        this.layers = new THREE.Layers();

        this.fk = {
            root: null,
            links: [],
            joints: [],
            axes: [],
            tcps: [],
        };

        this._simulator = physicsSimulator;
    }


    destroy() {
        for (const tool of this.tools) {
            if ((tool.button != null) && (tool.button.element != null))
                tool.button.element.remove();

            for (const mesh of tool.visual.meshes)
                mesh.layers = new THREE.Layers();
        }

        if (this.toolbar != null)
            this.toolbar.destroy();

        for (const segment of this.segments) {
            for (const mesh of segment.visual.meshes)
                mesh.layers = new THREE.Layers();

            for (const helper of segment.visual.helpers)
                helper.destroy();
        }
    }


    getMeshes() {
        return this.segments.map((segment) => segment.visual.meshes).flat().concat(
                    this.tools.map((tool) => tool.visual.meshes).flat()
        );
    }


    getJointPositions() {
        return this._simulator.getJointPositions(this.joints);
    }


    setJointPositions(positions) {
        const limits = this.segments.map((segment) => segment.limits).flat();

        const pos = positions.map(
            (v, i) => (Math.abs(limits[i][0]) > 1e-6) || (Math.abs(limits[i][1]) > 1e-6) ?
                            Math.min(Math.max(v, limits[i][0]), limits[i][1]) :
                            v
        );

        const nbJoints = positions.length;

        this._simulator.setJointPositions(pos, this.joints.slice(0, nbJoints));
        this._simulator.setControl(pos, this.actuators.slice(0, nbJoints));
    }


    getControl() {
        return this._simulator.getControl(this.actuators);
    }


    setControl(control) {
        const limits = this.segments.map((segment) => segment.limits).flat();

        const ctrl = control.map(
            (v, i) => (Math.abs(limits[i][0]) > 1e-6) || (Math.abs(limits[i][1]) > 1e-6) ?
                            Math.min(Math.max(v, limits[i][0]), limits[i][1]) :
                            v
        );

        const nbJoints = control.length;

        if (this._simulator.paused)
            this._simulator.setJointPositions(ctrl, this.joints.slice(0, nbJoints));

        this._simulator.setControl(ctrl, this.actuators.slice(0, nbJoints));
    }


    getJointVelocities() {
        return this._simulator.getJointVelocities(this.joints);
    }


    getCoM() {
        const bodyIdx = this.names.links.indexOf(this.configuration.robotRoot);
        return this._simulator.getSubtreeCoM(this.links[bodyIdx]);
    }


    getDefaultPose() {
        const pose = new Float32Array(this.joints.length);
        pose.fill(0.0);

        for (let name in this.configuration.defaultPose)
            pose[this.names.joints.indexOf(name)] = this.configuration.defaultPose[name];

        return pose;
    }


    applyDefaultPose() {
        this.setJointPositions(this.getDefaultPose());
    }


    getActuatorIndices(actuators) {
        return actuators.map((actuator) => this.actuators.indexOf(actuator));
    }


    /* Returns the number of end-effectors of the robot
    */
    getNbEndEffectors() {
        return this.tools.length;
    }


    /* Returns the position of a specific end-effector of the robot (a Vector3)
    */
    _getEndEffectorPosition(index=0) {
        this.tools[index].tcp.getWorldPosition(_tmpVector3$1);
        return _tmpVector3$1.clone();
    }


    /* Returns the orientation of a specific end-effector of the robot (a Quaternion)
    */
    _getEndEffectorOrientation(index=0) {
        this.tools[index].tcp.getWorldQuaternion(_tmpQuaternion$1);
        return _tmpQuaternion$1.clone();
    }


    /* Returns the position and orientation of a specific end-effector of the robot
    in an array of the form: [px, py, pz, qx, qy, qz, qw]
    */
    _getEndEffectorTransforms(index=0) {
        this.tools[index].tcp.getWorldPosition(_tmpVector3$1);
        this.tools[index].tcp.getWorldQuaternion(_tmpQuaternion$1);

        return [
            _tmpVector3$1.x, _tmpVector3$1.y, _tmpVector3$1.z,
            _tmpQuaternion$1.x, _tmpQuaternion$1.y, _tmpQuaternion$1.z, _tmpQuaternion$1.w,
        ];
    }


    /* Returns the desired position and orientation for a specific end-effector of
    the robot in an array of the form: [px, py, pz, qx, qy, qz, qw]

    The desired position and orientation are those of the manipulator of the
    end-effector (if enabled, see 'Viewer3D.endEffectorManipulation'), that the
    user can move freely.

    Returns:
        [px, py, pz, qx, qy, qz, qw]
    */
    _getEndEffectorDesiredTransforms(index=0) {
        if (this.tools[index].tcpTarget != null) {
            this.tools[index].tcpTarget.getWorldPosition(_tmpVector3$1);
            this.tools[index].tcpTarget.getWorldQuaternion(_tmpQuaternion$1);

            return [
                _tmpVector3$1.x, _tmpVector3$1.y, _tmpVector3$1.z,
                _tmpQuaternion$1.x, _tmpQuaternion$1.y, _tmpQuaternion$1.z, _tmpQuaternion$1.w,
            ];
        }

        return this._getEndEffectorTransforms(index);
    }


    /* Returns the position of all the end-effectors of the robot (an array of Nx3 values)
    */
    _getAllEndEffectorPositions() {
        const result = [];

        for (let tool of this.tools) {
            tool.tcp.getWorldPosition(_tmpVector3$1);
            result.push(_tmpVector3$1.x, _tmpVector3$1.y, _tmpVector3$1.z);
        }

        return result;
    }


    /* Returns the orientation of all the end-effectors of the robot (an array of Nx4 values,
    [qx, qy, qz, qw])
    */
    _getAllEndEffectorOrientations() {
        const result = [];

        for (let tool of this.tools) {
            tool.tcp.getWorldQuaternion(_tmpQuaternion$1);
            result.push(_tmpQuaternion$1.x, _tmpQuaternion$1.y, _tmpQuaternion$1.z, _tmpQuaternion$1.w);
        }

        return result;
    }


    /* Returns the position and orientation of all the end-effectors of the robot
    in an array of the form: N x [px, py, pz, qx, qy, qz, qw]
    */
    _getAllEndEffectorTransforms() {
        const result = [];

        for (let tool of this.tools) {
            tool.tcp.getWorldPosition(_tmpVector3$1);
            tool.tcp.getWorldQuaternion(_tmpQuaternion$1);

            result.push(
                _tmpVector3$1.x, _tmpVector3$1.y, _tmpVector3$1.z,
                _tmpQuaternion$1.x, _tmpQuaternion$1.y, _tmpQuaternion$1.z, _tmpQuaternion$1.w,
            );
        }

        return result;
    }


    /* Returns the desired position and orientation for all the end-effectors of
    the robot in an array of the form: N x [px, py, pz, qx, qy, qz, qw]

    The desired position and orientation are those of the manipulators of the
    end-effectors (if enabled, see 'Viewer3D.endEffectorManipulation'), that the
    user can move freely.

    Returns:
        [px, py, pz, qx, qy, qz, qw]
    */
    _getAllEndEffectorDesiredTransforms() {
        const result = [];

        for (let tool of this.tools) {
            if (tool.tcpTarget != null) {
                tool.tcpTarget.getWorldPosition(_tmpVector3$1);
                tool.tcpTarget.getWorldQuaternion(_tmpQuaternion$1);
            } else {
                tool.tcp.getWorldPosition(_tmpVector3$1);
                tool.tcp.getWorldQuaternion(_tmpQuaternion$1);
            }

            result.push(
                _tmpVector3$1.x, _tmpVector3$1.y, _tmpVector3$1.z,
                _tmpQuaternion$1.x, _tmpQuaternion$1.y, _tmpQuaternion$1.z, _tmpQuaternion$1.w,
            );
        }

        return result;
    }


    _enableTools(enabled, toolbar=null) {
        this.toolsEnabled = enabled && this.controlsEnabled;

        if (this.toolsEnabled) {
            this.toolbar = toolbar;

            if (toolbar != null) {
                for (let tool of this.tools) {
                    if (tool.button != null)
                        tool.button.object.visible = false;
                }
            }

            this._updateToolButtons();
        } else {
            for (let tool of this.tools) {
                if (tool.button != null)
                    tool.button.object.visible = false;
            }

            if (this.toolbar != null) {
                this.toolbar.destroy();
                this.toolbar = null;
            }
        }
    }


    _areToolsEnabled() {
        return this.toolsEnabled;
    }


    getKinematicChainForJoint(joint) {
        return new KinematicChain(this, joint);
    }


    getKinematicChainForTool(index=0) {
        return new KinematicChain(this, null, index);
    }


    _isGripperOpen(index=0) {
        const tool = this.tools[index];
        if (tool.type != 'gripper')
            return false;

        return (tool.state == 'opened') && (this.getGripperAbduction(index) >= 0.99);
    }


    _isGripperClosed(index=0) {
        const tool = this.tools[index];
        if (tool.type != 'gripper')
            return false;

        return (tool.state == 'closed') && (this.getGripperAbduction(index) <= 0.01);
    }


    _isGripperHoldingSomeObject(index=0) {
        const tool = this.tools[index];
        if (tool.type != 'gripper')
            return false;

        return (tool.state == 'closed') && (tool._stateCounter >= 5);
    }


    _getGripperAbduction(index=0) {
        const tool = this.tools[index];
        if (tool.type != 'gripper')
            return 0.0;

        if (tool.actuators.length === 0)
            return 0.0;

        // Average abduction of all tool joints
        let abduction = 0.0;
        let N = 0;
        const qpos = this._simulator.getJointPositions(tool.joints);

        for (let i = 0; i < tool.joints.length; ++i) {
            const joint = tool.joints[i];
            const actuator = this._simulator.getJointActuator(joint);

            if (tool.ignoredActuators.indexOf(actuator) >= 0)
                continue;

            const range = this._simulator.jointRange(joint);

            let rel = (qpos[i] - range[0]) / (range[1] - range[0]);

            if (tool.invertedActuators.indexOf(actuator) >= 0)
                rel = 1.0 - rel;

            abduction += rel;

            ++N;
        }
        abduction /= N;

        return abduction;
    }


    _closeGripper(index=0) {
        this._activateGripper(index, 'closed', 0);
    }


    _openGripper(index=0) {
        this._activateGripper(index, 'opened', 1);
    }


    _toggleGripper(index=0) {
        const tool = this.tools[index];
        if (tool.type != 'gripper')
            return;

        if (tool.state == 'opened')
            this._closeGripper(index);
        else
            this._openGripper(index);
    }


    createJointPositionHelpers(scene, layer, colors=[]) {
        const cfg = this.configuration.jointPositionHelpers;

        let index = 0;

        for (let segment of this.segments) {
            const x = this._simulator.getJointPositions(segment.joints);

            for (let i = 0; i < segment.joints.length; ++i) {
                const joint = segment.joints[i];
                const name = this._getJointName(joint);

                const axis = new THREE.Vector3();
                this._simulator._getPosition(this._simulator.model.jnt_axis, joint, axis);

                const helper = new JointPositionHelper(
                    scene, layer, joint, index + 1, axis, x[i],
                    cfg.inverted.includes(name),
                    colors[index] || 0xff0000,
                    cfg.offsets[name] || 0.0
                );

                helper.updateTransforms(segment.visual.joints[i]);

                segment.visual.helpers.push(helper);

                ++index;
            }
        }
    }


    synchronize(cameraPosition, elementWidth, tcpTarget=true) {
        // Note: The transforms of the visual representation of the links are already
        // updated by the physics simulator.

        // Update the robot joints visualisation (if necessary)
        for (let segment of this.segments) {
            const x = this._simulator.getJointPositions(segment.joints);

            for (let i = 0; i < segment.visual.helpers.length; ++i) {
                const helper = segment.visual.helpers[i];
                helper.updateTransforms(segment.visual.joints[i]);
                helper.updateJointPosition(x[i]);
                helper.updateSize(cameraPosition, elementWidth);
            }
        }

        const root = this.segments[0].visual.links[0];

        for (let i = 0; i < this.tools.length; ++i) {
            const tool = this.tools[i];

            // Synchronize the transforms of the TCP target element (if necessary)
            if (tcpTarget && (tool.tcpTarget != null)) {
                tool.tcp.getWorldPosition(_tmpVector3$1);
                tool.tcp.getWorldQuaternion(_tmpQuaternion$1);

                root.worldToLocal(_tmpVector3$1);
                tool.tcpTarget.position.copy(_tmpVector3$1);

                root.getWorldQuaternion(_tmpQuaternion2);
                tool.tcpTarget.quaternion.multiplyQuaternions(_tmpQuaternion2.invert(), _tmpQuaternion$1);
            }

            // Update the internal state of the tool
            if ((tool.type == 'gripper') && (tool.state == 'closed')) {
                const abduction = this.getGripperAbduction(i);
                if ((abduction > 0.01) && (Math.abs(abduction - tool._previousAbduction) < 1e-3)) {
                    tool._stateCounter++;

                    if (tool._stateCounter == 5) {
                        const pos = this._simulator.getJointPositions(tool.actuators);
                        this._simulator.setControl(pos, tool.actuators);
                    }
                } else {
                    tool._stateCounter = 0;
                    tool._previousAbduction = abduction;
                }
            }
        }

        // Update the buttons allowing to toggle the tools (if necessary)
        if (this.toolsEnabled)
            this._updateToolButtons();
    }


    _init() {
        // Retrieve the list of all joints, actuators and links
        for (let segment of this.segments) {
            this.joints = this.joints.concat(segment.joints);
            this.actuators = this.actuators.concat(segment.actuators);
            this.links = this.links.concat(segment.links);
        }

        // Retrieve the names of all joints, links and actuators
        this.names.joints = this._simulator.jointNames(this.joints);
        this.names.actuators = this._simulator.actuatorNames(this.actuators);
        this.names.links = this._simulator.bodyNames(this.links);

        for (let tool of this.tools) {
            tool.names.joints = this._simulator.jointNames(tool.joints);
            tool.names.actuators = this._simulator.actuatorNames(tool.actuators);
            tool.names.links = this._simulator.bodyNames(tool.links);
        }

        // Process each segment
        for (let segment of this.segments) {

            // Retrieve the limits of the joints of the segment
            for (let joint of segment.joints) {
                const actuator = this._simulator.getJointActuator(joint);
                let range = null;

                if (actuator != -1)
                    range = this._simulator.actuatorRange(actuator);
                else
                    range = this._simulator.jointRange(joint);

                segment.limits.push(range);
            }

            // Retrieve the list of all visual links (=groups) of the segment
            segment.visual.links = segment.links.map((b) => this._simulator.bodies[b]);

            // Retrieve the list of all visual links (=groups) with a joint of the segment
            segment.visual.joints = segment.joints.map((j) => this._simulator.bodies[this._simulator.model.jnt_bodyid[j]]);

            // Retrieve the list of all the meshes used by the segment
            segment.visual.meshes = segment.visual.links.map((body) => body.children.filter((c) => c.type == "Mesh")).flat();

            for (const mesh of segment.visual.meshes)
                mesh.layers = this.layers;
        }

        // Process each tool
        for (let i = 0; i < this.tools.length; ++i) {
            const tool = this.tools[i];

            // Retrieve the actuator values representing the states of the tools (if any)
            for (let actuator of tool.actuators) {
                const range = this._simulator.actuatorRange(actuator);

                if (tool.invertedActuators.indexOf(actuator) >= 0) {
                    tool.states.push({
                        closed: range[1],
                        opened: range[0],
                    });
                } else {
                    tool.states.push({
                        closed: range[0],
                        opened: range[1],
                    });
                }
            }

            // Retrieve the list of all visual links (=groups) of the tool
            tool.visual.links = tool.links.map((b) => this._simulator.bodies[b]);

            // Retrieve the list of all visual links (=groups) with a joint of the tool
            tool.visual.joints = tool.joints.map((j) => this._simulator.bodies[this._simulator.model.jnt_bodyid[j]]);

            // Retrieve the list of all the meshes used by the tool
            tool.visual.meshes = tool.visual.links.map((body) => body.children.filter((c) => c.type == "Mesh")).flat();

            for (const mesh of tool.visual.meshes)
                mesh.layers = this.layers;

            // Tool-specific actions (if necessary)
            if (tool.type == 'gripper') {
                // Initialise the internal state
                if (this.configuration.tools[i].state != null) {
                    tool.state = this.configuration.tools[i].state;
                } else {
                    if (this.getGripperAbduction(i) >= 0.99)
                        tool.state = 'opened';
                    else
                        tool.state = 'closed';
                }

                // Create the button to use the tool
                const img = document.createElement('img');
                if (tool.state == 'opened')
                    img.src = getURL('images/close_gripper.png');
                else
                    img.src = getURL('images/open_gripper.png');
                img.width = 24;
                img.height = 24;

                img.toolButtonFor = this;
                img.toolIndex = i;

                tool.button.element = document.createElement('div');
                tool.button.element.className = 'tool-button';
                tool.button.element.appendChild(img);

                tool.button.object = new CSS2DObject(tool.button.element);

                if (this.configuration.tools[i].buttonOffset != undefined)
                    tool.button.object.position.set(...this.configuration.tools[i].buttonOffset);

                tool.button.object.layers.disableAll();
                tool.button.object.layers.enable(31);

                tool.visual.links[0].add(tool.button.object);

                tool._previousAbduction = this.getGripperAbduction(i);
                tool._stateCounter = 0;
            }
        }

        if (this.tools.length == 1)
            this.kinematicChain = new KinematicChain(this, null, 0);
        else if (this.tools.length == 0)
            this.kinematicChain = new KinematicChain(this, this.joints[this.joints.length - 1]);

        // Create everything needed to do FK
        this._setupFK();

        // Apply the default pose defined in the configuration
        this.applyDefaultPose();
    }


    _activateGripper(index, stateName, rangeIndex) {
        const tool = this.tools[index];
        if (tool.type != 'gripper')
            return;

        if (tool.button.object != null)
            tool.button.object.visible = false;

        if (this.toolbar != null)
            this.toolbar.disable();

        if (tool.actuators.length - tool.ignoredActuators.length > 0) {
            const N = tool.actuators.length - tool.ignoredActuators.length;
            const ctrl = new Float32Array(N);
            const actuators = [];

            for (let i = 0, j = 0; i < tool.actuators.length; ++i) {
                const actuator = tool.actuators[i];
                if (tool.ignoredActuators.indexOf(actuator) >= 0)
                    continue;

                ctrl[j] = tool.states[i][stateName];
                actuators.push(actuator);

                ++j;
            }

            this._simulator.setControl(ctrl, actuators);
        }

        if (this._simulator.paused) {
            const start = {};
            const target = {};

            const qpos = this._simulator.getJointPositions(tool.joints);

            for (let i = 0; i < tool.joints.length; ++i) {
                const name = tool.names.joints[i];
                start[name] = qpos[i];
                target[name] = this._simulator.jointRange(tool.joints[i])[rangeIndex];
            }

            let tween = new TWEEN.Tween(start)
                .to(target, 500.0)
                .easing(TWEEN.Easing.Quadratic.Out);

            tween.onUpdate(object => {
                const x = new Float32Array(tool.joints.length);

                for (const name in object)
                    x[tool.names.joints.indexOf(name)] = object[name];

                this._simulator.setJointPositions(x, tool.joints);
            });

            tween.start();
        }

        tool.state = stateName;
        tool._stateCounter = 0;
    }


    _disableVisibility(materials) {
        for (const mesh of this.arm.visual.meshes) {
            const material = mesh.material;
            if (materials.indexOf(material) == -1) {
                material.colorWrite = false;
                material.depthWrite = false;
                materials.push(material);
            }
        }

        for (const mesh of this.tool.visual.meshes) {
            const material = mesh.material;
            if (materials.indexOf(material) == -1) {
                material.colorWrite = false;
                material.depthWrite = false;
                materials.push(material);
            }
        }

        for (const helper of this.arm.visual.helpers)
            helper._disableVisibility(materials);
    }


    _createTcpTarget() {
        const root = this.segments[0].visual.links[0];

        for (let i = 0; i < this.tools.length; ++i) {
            const tool = this.tools[i];

            let size = this.configuration.tools[i].tcpSize != undefined ? this.configuration.tools[i].tcpSize : 0.1;

            tool.tcpTarget = new THREE.Mesh(
                new THREE.SphereGeometry(size),
                new THREE.MeshBasicMaterial({
                    visible: false
                })
            );

            tool.tcpTarget.tag = 'tcp-target';
            tool.tcpTarget.robot = this;
            tool.tcpTarget.tool = i;

            tool.tcp.getWorldPosition(_tmpVector3$1);
            tool.tcp.getWorldQuaternion(_tmpQuaternion$1);

            root.worldToLocal(_tmpVector3$1);
            tool.tcpTarget.position.copy(_tmpVector3$1);

            root.getWorldQuaternion(_tmpQuaternion2);
            tool.tcpTarget.quaternion.multiplyQuaternions(_tmpQuaternion2.invert(), _tmpQuaternion$1);

            root.add(tool.tcpTarget);
        }
    }


    _setupFK() {
        this.segments[0].visual.links[0].getWorldPosition(_tmpVector3$1);
        this.segments[0].visual.links[0].getWorldQuaternion(_tmpQuaternion$1);

        this.fk.root = new THREE.Object3D();
        this.fk.root.position.copy(_tmpVector3$1);
        this.fk.root.quaternion.copy(_tmpQuaternion$1);
        this.fk.root.bodyId = this.segments[0].visual.links[0].bodyId;
        this.fk.root.name = this.segments[0].visual.links[0].name;
        this.fk.links.push(this.fk.root);

        let allLinks = this.segments.map((segment) => segment.visual.links);
        allLinks = allLinks.concat(this.tools.map((tool) => tool.visual.links));
        allLinks = allLinks.flat();

        for (let i = 1; i < allLinks.length; ++i) {
            const ref = allLinks[i];

            let parent = null;

            if (ref.jointId != undefined) {
                const joint = new THREE.Object3D();
                joint.jointId = ref.jointId;
                joint.axis = new THREE.Vector3();
                joint.name = this._simulator.names[this._simulator.model.name_jntadr[ref.jointId]];

                this._simulator._getPosition(this._simulator.model.jnt_pos, ref.jointId, joint.position);
                joint.position.add(ref.position);

                this._simulator._getPosition(this._simulator.model.jnt_axis, ref.jointId, joint.axis);
                joint.quaternion.setFromAxisAngle(joint.axis, 0.0);
                joint.quaternion.premultiply(ref.quaternion);

                joint.refQuaternion = ref.quaternion.clone();
                const parentBody = this.fk.links.filter((l) => l.bodyId == ref.parent.bodyId)[0];
                parentBody.add(joint);

                this.fk.joints.push(joint);

                joint.updateMatrixWorld(true);

                parent = joint;

            } else {
                parent = this.fk.links.filter((l) => l.bodyId == ref.parent.bodyId)[0];
            }

            const link = new THREE.Object3D();
            link.bodyId = ref.bodyId;
            link.name = ref.name;

            if (ref.jointId == undefined) {
                link.position.copy(ref.position);
                link.quaternion.copy(ref.quaternion);
            } else {
                this._simulator._getPosition(this._simulator.model.jnt_pos, ref.jointId, _tmpVector3$1);
                link.position.sub(_tmpVector3$1);
            }

            parent.add(link);

            link.updateMatrixWorld(true);

            this.fk.links.push(link);
        }

        for (let tool of this.tools) {
            const tcp = new THREE.Object3D();
            tcp.position.copy(tool.tcp.position);
            tcp.quaternion.copy(tool.tcp.quaternion);

            const parent = this.fk.links.filter((l) => l.bodyId == tool.tcp.parent.bodyId)[0];
            parent.add(tcp);

            this.fk.tcps.push(tcp);
        }
    }


    _setJointPosition(jointId, position) {
        const joint = this.fk.joints.filter((j) => j.jointId == jointId)[0];

        joint.quaternion.setFromAxisAngle(joint.axis, position);
        joint.quaternion.premultiply(joint.refQuaternion);
        joint.matrixWorldNeedsUpdate = true;
    }


    _updateToolButtons() {
        if (this.toolbar != null) {
            if (!this.toolbar.isEnabled()) {
                for (let i = 0; i < this.tools.length; ++i) {
                    const tool = this.tools[i];
                    if (tool.type == 'gripper') {
                        if (this.isGripperOpen(i)) {
                            this.toolbar.update(false);
                        } else if (this.isGripperClosed(i) || this.isGripperHoldingSomeObject(i)) {
                            this.toolbar.update(true);
                        }
                    }
                }
            }

        } else {
            for (let i = 0; i < this.tools.length; ++i) {
                const tool = this.tools[i];

                 if ((tool.button != undefined) && (tool.button.object != null)) {
                     if (!tool.button.object.visible) {
                        if (tool.type == 'gripper') {
                            if (this.isGripperOpen(i)) {
                                tool.button.element.children[0].src = getURL('images/close_gripper.png');
                                tool.button.object.visible = true;

                            } else if (this.isGripperClosed(i) || this.isGripperHoldingSomeObject(i)) {
                                tool.button.element.children[0].src = getURL('images/open_gripper.png');
                                tool.button.object.visible = true;
                            }
                        }
                    }
                }
            }
        }
    }


    _createSegment(body, parent=null) {
        const segment = {
            joints: [],
            actuators: [],
            links: [body],
            limits: [],

            visual: {
                joints: [],
                links: [],
                meshes: [],
                helpers: [],
            },

            parent: this.segments.indexOf(parent),
        };

        this.segments.push(segment);

        return segment;
    }


    _createTool(tcp, body, configuration) {
        let tool = {
            type: configuration.type,

            joints: [],
            actuators: [],
            links: [],

            tcp: tcp,
            tcpTarget: null,

            names: {
                joints: [],
                actuators: [],
                links: [],
            },

            visual: {
                joints: [],
                links: [],
                meshes: [],
            },

            parent: -1,
        };

        if (body != null)
            tool.links.push(body);

        if (configuration.type == "gripper") {
            const additions = {
                states: [],
                state: null,

                ignoredActuators: [],
                invertedActuators: [],

                button: {
                    object: null,
                    element: null,
                },

                _previousAbduction: null,
                _stateCounter: 0,
            };

            tool = {...tool, ...additions};
        }

        this.tools.push(tool);

        return tool;
    }


    _getJointName(joint) {
        const idx = this.joints.indexOf(joint);
        if (idx >= 0)
            return this.names.joints[idx];

        return null;
    }


    _getSegmentOfJoint(joint) {
        for (let segment of this.segments) {
            if (segment.joints.indexOf(joint) != -1)
                return segment;
        }

        return null;
    }

    _getSegmentOfBody(body) {
        for (let segment of this.segments) {
            if (segment.links.indexOf(body) != -1)
                return segment;
        }

        return null;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */




class SimpleRobot extends Robot {

    constructor(name, configuration, physicsSimulator) {
        super(name, configuration, physicsSimulator);

        this.kinematicChain = null;
    }


    /* Returns the position of the end-effector of the robot (a Vector3)
    */
    getEndEffectorPosition() {
        return this._getEndEffectorPosition();
    }


    /* Returns the orientation of the end-effector of the robot (a Quaternion)
    */
    getEndEffectorOrientation() {
        return this._getEndEffectorOrientation();
    }


    /* Returns the position and orientation of the end-effector of the robot
    in an array of the form: [px, py, pz, qx, qy, qz, qw]
    */
    getEndEffectorTransforms() {
        return this._getEndEffectorTransforms();
    }


    /* Returns the desired position and orientation for the end-effector of
    the robot in an array of the form: [px, py, pz, qx, qy, qz, qw]

    The desired position and orientation are those of the manipulator of the
    end-effector (if enabled, see 'Viewer3D.endEffectorManipulation'), that the
    user can move freely.

    Returns:
        [px, py, pz, qx, qy, qz, qw]
    */
    getEndEffectorDesiredTransforms() {
        return this._getEndEffectorDesiredTransforms();
    }


    enableTool(enabled, toolbar=null) {
        this._enableTools(enabled, toolbar);
    }


    isToolEnabled() {
        return this._areToolsEnabled();
    }


    /* Performs Forward Kinematics, on a subset of the joints

    Parameters:
        positions (array): The joint positions
        offset (Vector3): Optional, an offset from the last joint

    Returns:
        A tuple of a Vector3 and a Quaternion: (position, orientation)
    */
    fkin(positions, offset=null) {
        return this.kinematicChain.fkin(positions, offset);
    }


    ik(mu, nbJoints=null, offset=null, limit=5, dt=0.01, successDistance=1e-4, damping=false) {
        return this.kinematicChain.ik(mu, nbJoints, offset, limit, dt, successDistance, damping);
    }


    /* Jacobian with numerical computation, on a subset of the joints
    */
    Jkin(positions, offset=null) {
        return this.kinematicChain.Jkin(positions, offset);
    }


    isGripperOpen() {
        return this._isGripperOpen();
    }


    isGripperClosed() {
        return this._isGripperClosed();
    }


    isGripperHoldingSomeObject() {
        return this._isGripperHoldingSomeObject();
    }


    getGripperAbduction() {
        return this._getGripperAbduction();
    }


    closeGripper() {
        this._closeGripper();
    }


    openGripper() {
        this._openGripper();
    }


    toggleGripper() {
        this._toggleGripper();
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



const _tmpVector3 = new THREE.Vector3();
const _tmpQuaternion = new THREE.Quaternion();


class ComplexRobot extends Robot {

    constructor(name, configuration, physicsSimulator) {
        super(name, configuration, physicsSimulator);
    }


    /* Returns the position of a specific end-effector of the robot (a Vector3)
    */
    getEndEffectorPosition(index=0) {
        return this._getEndEffectorPosition(index);
    }


    /* Returns the orientation of a specific end-effector of the robot (a Quaternion)
    */
    getEndEffectorOrientation(index=0) {
        return this._getEndEffectorOrientation(index);
    }


    /* Returns the position and orientation of a specific end-effector of the robot
    in an array of the form: [px, py, pz, qx, qy, qz, qw]
    */
    getEndEffectorTransforms(index=0) {
        return this._getEndEffectorTransforms(index);
    }


    /* Returns the desired position and orientation for a specific end-effector of
    the robot in an array of the form: [px, py, pz, qx, qy, qz, qw]

    The desired position and orientation are those of the manipulator of the
    end-effector (if enabled, see 'Viewer3D.endEffectorManipulation'), that the
    user can move freely.

    Returns:
        [px, py, pz, qx, qy, qz, qw]
    */
    getEndEffectorDesiredTransforms(index=0) {
        return this._getEndEffectorDesiredTransforms(index);
    }


    /* Returns the position of all the end-effectors of the robot (an array of Nx3 values)
    */
    getAllEndEffectorPositions() {
        return this._getAllEndEffectorPositions();
    }


    /* Returns the orientation of all the end-effectors of the robot (an array of Nx4 values,
    [qx, qy, qz, qw])
    */
    getAllEndEffectorOrientations() {
        return this._getAllEndEffectorOrientations();
    }


    /* Returns the position and orientation of all the end-effectors of the robot
    in an array of the form: N x [px, py, pz, qx, qy, qz, qw]
    */
    getAllEndEffectorTransforms() {
        return this._getAllEndEffectorTransforms();
    }


    /* Returns the desired position and orientation for all the end-effectors of
    the robot in an array of the form: N x [px, py, pz, qx, qy, qz, qw]

    The desired position and orientation are those of the manipulators of the
    end-effectors (if enabled, see 'Viewer3D.endEffectorManipulation'), that the
    user can move freely.

    Returns:
        [px, py, pz, qx, qy, qz, qw]
    */
    getAllEndEffectorDesiredTransforms() {
        return this._getAllEndEffectorDesiredTransforms();
    }


    enableTools(enabled, toolbar=null) {
        this._enableTools(enabled, toolbar);
    }


    areToolsEnabled() {
        return this._areToolsEnabled();
    }


    isGripperOpen(index=0) {
        return this._isGripperOpen(index);
    }


    isGripperClosed(index=0) {
        return this._isGripperClosed(index);
    }


    isGripperHoldingSomeObject(index=0) {
        this._isGripperHoldingSomeObject(index);
    }


    getGripperAbduction(index=0) {
        return this._getGripperAbduction(index);
    }


    closeGripper(index=0) {
        this._closeGripper(index);
    }


    openGripper(index=0) {
        this._openGripper(index);
    }


    toggleGripper(index=0) {
        this._toggleGripper(index);
    }


    fkin(positions) {
        if (math.typeOf(positions) == 'DenseMatrix')
            positions = positions.toArray();
        else if (math.typeOf(positions) == 'number')
            positions = [positions];

        // Check the input
        if (positions.length != this.joints.length)
            throw new Error('The number of joint positions must be equal to the number of movable joints');

        // Set the joint positions
        const nbJoints = positions.length;

        for (let i = 0; i < nbJoints; ++i)
            this._setJointPosition(this.joints[i], positions[i]);

        // Retrieve the position and orientation of the end-effectors
        const result = [];

        for (let tcp of this.fk.tcps) {
            tcp.getWorldPosition(_tmpVector3);
            tcp.getWorldQuaternion(_tmpQuaternion);

            result.push(
                _tmpVector3.x, _tmpVector3.y, _tmpVector3.z,
                _tmpQuaternion.x, _tmpQuaternion.y, _tmpQuaternion.z, _tmpQuaternion.w
            );
        }

        return result;
    }


    Jkin(positions) {
        const eps = 1e-6;

        if (math.typeOf(positions) == 'Array')
            positions = math.matrix(positions);
        else if (math.typeOf(positions) == 'number')
            positions = math.matrix([ positions ]);

        const D = positions.size()[0];
        const N = this.tools.length * 7;

        positions = math.reshape(positions, [D, 1]);

        // Matrix computation
        const F1 = math.zeros(N, D);
        const F2 = math.zeros(N, D);

        const f0 = math.matrix(this.fkin(positions));

        for (let i = 0; i < D; ++i) {
            F1.subset(math.index(math.range(0, N), i), f0);

            const diff = math.zeros(D);
            diff.set([i, 0], eps);

            F2.subset(math.index(math.range(0, N), i), this.fkin(math.add(positions, diff)));
        }

        let J = math.matrix(this.tools.length * 6, D);

        for (let i = 0; i < this.tools.length; ++i) {
            const Findices = math.index(math.range(i * 7, (i + 1) * 7), math.range(0, D));
            const Jindices = math.index(math.range(i * 6, (i + 1) * 6), math.range(0, D));

            J.subset(Jindices, math.divide(logmap(F2.subset(Findices), F1.subset(Findices)), eps));
        }

        if (J.size().length == 1)
            J = math.reshape(J, [J.size()[0], 1]);

        return J;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */




function loadScene(filename, robotBuilders=null) {
    // Retrieve some infos from the XML file (not exported by the MuJoCo API)
    const xmlDoc = loadXmlFile(filename);
    if (xmlDoc == null)
        return null;

    filename += ".processed";

    const serializer = new XMLSerializer();
    mujoco.FS.writeFile(filename, serializer.serializeToString(xmlDoc));

    const freeCameraSettings = getFreeCameraSettings(xmlDoc);
    const statistics = getStatistics(xmlDoc);
    const fogSettings = getFogSettings(xmlDoc);
    const headlightSettings = getHeadlightSettings(xmlDoc, filename);
    const lightIntensities = getLightIntensities(xmlDoc, filename);

    // Add the robot if a builder was provided
    if ((robotBuilders != null) && (robotBuilders.length > 0)) {
        for (const builder of robotBuilders)
            generateRobot(xmlDoc, filename, builder);
    }

    // Preprocess the included files if necessary
    let includedLightIntensities = preprocessIncludedFiles(xmlDoc, filename, (robotBuilders != null) && (robotBuilders.length > 0));

    includedLightIntensities = includedLightIntensities.concat(lightIntensities);

    // Load in the state from XML
    let model = new mujoco.Model(filename);

    return new PhysicsSimulator(
        model, freeCameraSettings, statistics, fogSettings, headlightSettings, includedLightIntensities
    );
}



class PhysicsSimulator {

    constructor(model, freeCameraSettings, statistics, fogSettings, headlightSettings, lightIntensities) {
        this.model = model;
        this.state = new mujoco.State(model);
        this.simulation = new mujoco.Simulation(model, this.state);

        this.freeCameraSettings = freeCameraSettings;
        this.statistics = null;
        this.fogSettings = fogSettings;
        this.headlightSettings = headlightSettings;

        // Initialisations
        this.bodies = {};
        this.meshes = {};
        this.textures = {};
        this.lights = [];
        this.ambientLight = null;
        this.headlight = null;
        this.sites = {};
        this.infinitePlanes = [];
        this.infinitePlane = null;
        this.paused = true;
        this.time = 0.0;

        // Decode the null-terminated string names
        this.names = {};

        const textDecoder = new TextDecoder("utf-8");
        const fullString = textDecoder.decode(model.names);

        let start = 0;
        let end = fullString.indexOf('\0', start);
        while (end != -1) {
            this.names[start] = fullString.substring(start, end);
            start = end + 1;
            end = fullString.indexOf('\0', start);
        }

        // Create a list of all joints not used by a robot (will be modified each time
        // a robot is declared)
        this.freeJoints = [];
        for (let j = 0; j < this.model.njnt; ++j)
            this.freeJoints.push(j);

        // Create the root object
        this.root = new THREE.Group();
        this.root.name = "MuJoCo Root";

        // Process the elements
        this._processGeometries();
        this._processLights(lightIntensities);
        this._processSites();

        // Ensure each body controlled by a joint knows the joint ID
        for (let j = 0; j < this.model.njnt; ++j) {
            const bodyId = this.model.jnt_bodyid[j];
            this.bodies[bodyId].jointId = j;
        }

        // Compute informations like MuJoCo does
        this.simulation.forward();

        this._computeStatistics(statistics);

        const scale = 2.0 * this.freeCameraSettings.zfar * this.statistics.extent;
        for (const mesh of this.infinitePlanes) {
            mesh.scale.set(mesh.infiniteX ? scale : 1.0, mesh.infiniteY ? scale : 1.0, 1.0);

            if (mesh.texuniform) {
                if (mesh.infiniteX)
                    mesh.material.map.repeat.x *= scale;

                if (mesh.infiniteY)
                    mesh.material.map.repeat.y *= scale;
            }
        }
        delete this.infinitePlanes;
    }


    destroy() {
        this.simulation.delete();
        this.state.delete();
        this.model.delete();
    }


    update(time) {
        if (!this.paused) {
            let timestep = this.model.getOptions().timestep;

            if (time - this.time > 0.035)
                timestep *= 2;

            while (this.time < time) {
                this.simulation.step();
                this.time += timestep;
            }
        } else {
            this.simulation.forward();
        }

        for (let i = 0; i < this.model.nbody; ++i) {
            this.simulation.xfrc_applied[i * 6] = 0.0;
            this.simulation.xfrc_applied[i * 6 + 1] = 0.0;
            this.simulation.xfrc_applied[i * 6 + 2] = 0.0;
            this.simulation.xfrc_applied[i * 6 + 3] = 0.0;
            this.simulation.xfrc_applied[i * 6 + 4] = 0.0;
            this.simulation.xfrc_applied[i * 6 + 5] = 0.0;
        }
    }


    synchronize() {
        // Update body transforms
        const pos = new THREE.Vector3();
        const orient1 = new THREE.Quaternion();
        const orient2 = new THREE.Quaternion();

        for (let b = 1; b < this.model.nbody; ++b) {
            const body = this.bodies[b];
            const parent_body_id = this.model.body_parentid[b];

            if (parent_body_id > 0) {
                const parent_body = this.bodies[parent_body_id];

                this._getPosition(this.simulation.xpos, b, pos);
                this._getQuaternion(this.simulation.xquat, b, orient2);

                parent_body.worldToLocal(pos);
                body.position.copy(pos);

                parent_body.getWorldQuaternion(orient1);
                orient1.invert();

                body.quaternion.multiplyQuaternions(orient1, orient2);
            } else {
                this._getPosition(this.simulation.xpos, b, body.position);
                this._getQuaternion(this.simulation.xquat, b, body.quaternion);
            }

            body.updateWorldMatrix();
        }

        // Update light transforms
        const dir = new THREE.Vector3();
        for (let l = 0; l < this.model.nlight; ++l) {
            if (this.lights[l]) {
                const light = this.lights[l];

                this._getPosition(this.simulation.light_xpos, l, pos);
                this._getPosition(this.simulation.light_xdir, l, dir);

                light.target.position.copy(dir.add(pos));

                light.parent.worldToLocal(pos);
                light.position.copy(pos);
            }
        }
    }


    bodyNames(indices=null) {
        const names = [];

        if (indices == null) {
            for (let b = 0; b < this.model.nbody; ++b)
                names.push(this.names[this.model.name_bodyadr[b]]);
        } else {
            for (let i = 0; i < indices.length; ++i) {
                const b = indices[i];
                names.push(this.names[this.model.name_bodyadr[b]]);
            }
        }

        return names;
    }


    jointNames(indices=null) {
        const names = [];

        if (indices == null) {
            for (let j = 0; j < this.model.njnt; ++j)
                names.push(this.names[this.model.name_jntadr[j]]);
        } else {
            for (let i = 0; i < indices.length; ++i) {
                const j = indices[i];
                names.push(this.names[this.model.name_jntadr[j]]);
            }
        }

        return names;
    }


    actuatorNames(indices=null) {
        const names = [];

        if (indices == null) {
            for (let a = 0; a < this.model.nu; ++a)
                names.push(this.names[this.model.name_actuatoradr[a]]);
        } else {
            for (let i = 0; i < indices.length; ++i) {
                const a = indices[i];
                names.push(this.names[this.model.name_actuatoradr[a]]);
            }
        }

        return names;
    }


    jointRange(jointId) {
        return this.model.jnt_range.slice(jointId * 2, jointId * 2 + 2);
    }


    actuatorRange(actuatorId) {
        return this.model.actuator_ctrlrange.slice(actuatorId * 2, actuatorId * 2 + 2);
    }


    getJointActuator(jointId) {
        const index = this.model.actuator_trnid.indexOf(jointId);
        if (index != -1)
            return index  / 2;

        return -1;
    }


    getJointPositions(indices=null) {
        if (indices == null)
            return new Float64Array(this.simulation.qpos);

        const qpos = new Float64Array(indices.length);

        for (let i = 0; i < indices.length; ++i) {
            const j = indices[i];
            qpos[i] = this.simulation.qpos[this.model.jnt_qposadr[j]];
        }

        return qpos;
    }


    setJointPositions(positions, indices=null) {
        if (indices == null) {
            this.simulation.qpos.set(positions);

        } else {
            for (let i = 0; i < indices.length; ++i) {
                const j = indices[i];
                this.simulation.qpos[this.model.jnt_qposadr[j]] = positions[i];
            }
        }
    }


    getControl(indices=null) {
        if (indices == null)
            return new Float64Array(this.simulation.ctrl);

        const ctrl = new Float64Array(indices.length);

        for (let i = 0; i < indices.length; ++i) {
            const a = indices[i];
            ctrl[i] = this.simulation.ctrl[a];
        }

        return ctrl;
    }


    setControl(ctrl, indices=null) {
        if (indices == null) {
            this.simulation.ctrl.set(ctrl);

        } else {
            for (let i = 0; i < indices.length; ++i) {
                const a = indices[i];
                this.simulation.ctrl[a] = ctrl[i];
            }
        }
    }


    getJointVelocities(indices=null) {
        if (indices == null)
            return new Float64Array(this.simulation.qvel);

        const qvel = new Float64Array(indices.length);

        for (let i = 0; i < indices.length; ++i) {
            const j = indices[i];
            qvel[i] = this.simulation.qvel[this.model.jnt_qposadr[j]];
        }

        return qvel;
    }


    getBodyId(name) {
        for (let b = 0; b < this.model.nbody; ++b) {
            const bodyName = this.names[this.model.name_bodyadr[b]];

            if (bodyName == name)
                return b;
        }

        return null;
    }


    getBodyPosition(bodyId) {
        const pos = new THREE.Vector3();
        this._getPosition(this.simulation.xpos, bodyId, pos);
        return pos;
    }


    setBodyPosition(bodyId, position) {
        const jntadr = this.model.body_jntadr[bodyId];
        this.simulation.qpos[jntadr] = position.x;
        this.simulation.qpos[jntadr + 1] = position.y;
        this.simulation.qpos[jntadr + 2] = position.z;
    }


    getBodyOrientation(bodyId) {
        const quat = new THREE.Quaternion();
        this._getQuaternion(this.simulation.xquat, bodyId, quat);
        return quat;
    }


    setBodyOrientation(bodyId, orientation) {
        const jntadr = this.model.body_jntadr[bodyId] + 3;
        this.simulation.qpos[jntadr] = orientation.w;
        this.simulation.qpos[jntadr + 1] = orientation.x;
        this.simulation.qpos[jntadr + 2] = orientation.y;
        this.simulation.qpos[jntadr + 3] = orientation.z;
    }


    getSubtreeCoM(bodyId) {
        const pos = new THREE.Vector3();
        this._getPosition(this.simulation.subtree_com, bodyId, pos);
        return pos;
    }


    createRobot(name, configuration, prefix=null) {
        const sim = this;

        function _getChildBodies(bodyIdx, children) {
            for (let b = bodyIdx + 1; b < sim.model.nbody; ++b) {
                if (sim.names[sim.model.name_bodyadr[b]] == configuration.toolRoot)
                    continue;

                if (sim.model.body_parentid[b] == bodyIdx) {
                    children.push(b);
                    _getChildBodies(b, children);
                }
            }
        }

        function _getDirectChildBodies(bodyIdx) {
            const children = [];

            for (let b = bodyIdx + 1; b < sim.model.nbody; ++b) {
                if (sim.model.body_parentid[b] == bodyIdx)
                    children.push(b);
            }

            return children;
        }

        function _buildSegments(bodyIdx, segment, toolBodies) {
            const children =  _getDirectChildBodies(bodyIdx);

            if (children.length == 1) {
                if (toolBodies.indexOf(children[0]) == -1) {
                    segment.links.push(children[0]);
                    _buildSegments(children[0], segment, toolBodies);
                }

            } else if (children.length > 1) {
                for (let i = 0; i < children.length; ++i) {
                    if (toolBodies.indexOf(children[i]) == -1) {
                        const segment2 = robot._createSegment(children[i], segment);
                        _buildSegments(children[i], segment2, toolBodies);
                    }
                }
            }
        }

        function _getBody(name) {
            for (let b = 0; b < sim.model.nbody; ++b) {
                const bodyName = sim.names[sim.model.name_bodyadr[b]];

                if (bodyName == name)
                    return b;
            }

            return null;
        }

        function _getSite(name) {
            for (let s = 0; s < sim.model.nsite; ++s) {
                const siteName = sim.names[sim.model.name_siteadr[s]];

                if (siteName == name)
                    return sim.sites[s];
            }

            return null;
        }


        // Modify the configuration if a prefix was provided
        if (prefix != null)
            configuration = configuration.addPrefix(prefix);

        // Create the robot
        const robot = configuration.tools.length > 1 ?
                        new ComplexRobot(name, configuration, this) :
                        new SimpleRobot(name, configuration, this);

        // Retrieve all the tools of the robot
        const toolBodies = [];
        for (let j = 0; j < configuration.tools.length; ++j) {
            let body = null;
            if (configuration.tools[j].root != null)
            {
                body = _getBody(configuration.tools[j].root);
                if (body == null) {
                    console.error("Failed to create the robot: link '" + configuration.tools[j].root + "' not found");
                    return null;
                }

                toolBodies.push(body);
            }

            const tcp = _getSite(configuration.tools[j].tcpSite);

            const tool = robot._createTool(tcp, body, configuration.tools[j]);

            if (body != null)
                _getChildBodies(body, tool.links);
        }

        // Retrieve all the segments of the robot
        const rootBody = _getBody(configuration.robotRoot);
        if (rootBody == null) {
            console.error("Failed to create the robot: link '" + configuration.robotRoot + "' not found");
            return null;
        }

        const segment = robot._createSegment(rootBody);
        _buildSegments(rootBody, segment, toolBodies);

        // Retrieve all the joints of the robot
        for (let j = 0; j < this.model.njnt; ++j) {
            const type = this.model.jnt_type[j];
            if (type == mujoco.mjtJoint.mjJNT_FREE.value)
                continue;

            const body = this.model.jnt_bodyid[j];

            let found = false;
            for (let segment of robot.segments) {
                if (segment.links.indexOf(body) >= 0) {
                    segment.joints.push(j);
                    this.freeJoints.splice(this.freeJoints.indexOf(j), 1);
                    found = true;
                    break;
                }
            }

            if (found)
                continue;

            for (let tool of robot.tools) {
                if (tool.links.indexOf(body) >= 0) {
                    tool.joints.push(j);
                    this.freeJoints.splice(this.freeJoints.indexOf(j), 1);
                    break;
                }
            }
        }

        // Merge segments
        let modified = true;
        while (modified) {
            modified = false;

            for (let j = robot.segments.length - 1; j >= 0; --j) {
                const segment = robot.segments[j];
                if ((segment.joints.length == 0) && (segment.parent >= 0)) {
                    const parent = robot.segments[segment.parent];
                    parent.links = parent.links.concat(segment.links);

                    robot.segments.splice(j, 1);

                    for (let k = 0; k < robot.segments.length; ++k) {
                        if (robot.segments[k].parent >= j)
                            robot.segments[k].parent -= 1;
                    }

                    modified = true;
                    break;
                }
            }
        }

        // Retrieve all the actuators of the robot
        for (let a = 0; a < this.model.nu; ++a) {
            const type = this.model.actuator_trntype[a];
            const id = this.model.actuator_trnid[a * 2];

            if ((type == mujoco.mjtTrn.mjTRN_JOINT.value) ||
                (type == mujoco.mjtTrn.mjTRN_JOINTINPARENT.value)) {

                let found = false;
                for (let segment of robot.segments) {
                    if (segment.joints.indexOf(id) >= 0) {
                        segment.actuators.push(a);
                        found = true;
                        break;
                    }
                }

                if (found)
                    continue;

                for (let j = 0; j < robot.tools.length; ++j) {
                    const tool = robot.tools[j];

                    if (tool.joints.indexOf(id) >= 0) {
                        tool.actuators.push(a);

                        const cfg = configuration.tools[j];

                        if ((cfg.ignoredActuators != undefined) && (cfg.ignoredActuators.indexOf(this.actuatorNames([a])[0]) >= 0))
                            tool.ignoredActuators.push(a);

                        if ((cfg.invertedActuators != undefined) && (cfg.invertedActuators.indexOf(this.actuatorNames([a])[0]) >= 0))
                            tool.invertedActuators.push(a);

                        break;
                    }
                }

            } else if (type == mujoco.mjtTrn.mjTRN_TENDON.value) {
                const adr = this.model.tendon_adr[id];
                const nb = this.model.tendon_num[id];

                for (let w = adr; w < adr + nb; ++w) {
                    if (this.model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_JOINT.value) {
                        const jointId = this.model.wrap_objid[w];

                        let found = false;
                        for (let segment of robot.segments) {
                            if (segment.joints.indexOf(jointId) >= 0) {
                                segment.actuators.push(a);
                                found = true;
                                break;
                            }
                        }

                        if (found)
                            continue;

                        for (let j = 0; j < robot.tools.length; ++j) {
                            const tool = robot.tools[j];

                            if (tool.joints.indexOf(jointId) >= 0) {
                                tool.actuators.push(a);

                                const cfg = configuration.tools[j];

                                if ((cfg.ignoredActuators != undefined) && (cfg.ignoredActuators.indexOf(this.actuatorNames([a])[0]) >= 0))
                                    tool.ignoredActuators.push(a);

                                if ((cfg.invertedActuators != undefined) && (cfg.invertedActuators.indexOf(this.actuatorNames([a])[0]) >= 0))
                                    tool.invertedActuators.push(a);

                                break;
                            }
                        }

                        break;
                    }
                }
            }
        }

        // Retrieve the parent segment of each tool
        for (let j = 0; j < robot.tools.length; ++j) {
            const tool = robot.tools[j];

            let parentBody = null;
            if (configuration.tools[j].root != null)
                parentBody = this.model.body_parentid[tool.links[0]];
            else
                parentBody = this.model.site_bodyid[tool.tcp.site_id];

            tool.parent = robot._getSegmentOfBody(parentBody);
        }

        // Let the robot initialise its internal state
        robot._init();

        return robot;
    }


    getBackgroundTextures() {
        for (let t = 0; t < this.model.ntex; ++t) {
            if (this.model.tex_type[t] == mujoco.mjtTexture.mjTEXTURE_SKYBOX.value)
                return this._createTexture(t);
        }

        return null;
    }


    _processGeometries() {
        // Default material definition
        const defaultMaterial = new THREE.MeshPhysicalMaterial();
        defaultMaterial.color = new THREE.Color(1, 1, 1);

        // Loop through the MuJoCo geoms and recreate them in three.js
        for (let g = 0; g < this.model.ngeom; g++) {
            // Only visualize geom groups up to 2
            if (!(this.model.geom_group[g] < 3)) {
                continue;
            }

            // Get the body ID and type of the geom
            let b = this.model.geom_bodyid[g];
            let type = this.model.geom_type[g];
            let size = [
                this.model.geom_size[(g * 3) + 0],
                this.model.geom_size[(g * 3) + 1],
                this.model.geom_size[(g * 3) + 2]
            ];

            // Create the body if it doesn't exist
            if (!(b in this.bodies)) {
                this.bodies[b] = new THREE.Group();
                this.bodies[b].name = this.names[this.model.name_bodyadr[b]];
                this.bodies[b].bodyId = b;
                this.bodies[b].has_custom_mesh = false;
            }

            // Set the default geometry (in MuJoCo, this is a sphere)
            let geometry = new THREE.SphereGeometry(size[0] * 0.5);
            if (type == mujoco.mjtGeom.mjGEOM_PLANE.value) ; else if (type == mujoco.mjtGeom.mjGEOM_HFIELD.value) ; else if (type == mujoco.mjtGeom.mjGEOM_SPHERE.value) {
                geometry = new THREE.SphereGeometry(size[0]);
            } else if (type == mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
                geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 20, 20);
            } else if (type == mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
                geometry = new THREE.SphereGeometry(1); // Stretch this below
            } else if (type == mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
                geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0, 20);
            } else if (type == mujoco.mjtGeom.mjGEOM_BOX.value) {
                geometry = new THREE.BoxGeometry(size[0] * 2.0, size[1] * 2.0, size[2] * 2.0);
            } else if (type == mujoco.mjtGeom.mjGEOM_MESH.value) {
                let meshID = this.model.geom_dataid[g];

                if (!(meshID in this.meshes)) {
                    geometry = new THREE.BufferGeometry();

                    // Positions
                    let vertices = this.model.mesh_vert.subarray(
                        this.model.mesh_vertadr[meshID] * 3,
                        (this.model.mesh_vertadr[meshID] + this.model.mesh_vertnum[meshID]) * 3
                    );

                    const vertex_buffer = new Float32Array(this.model.mesh_facenum[meshID] * 3 * 3);

                    // Normals
                    let normals = this.model.mesh_normal.subarray(
                        this.model.mesh_normaladr[meshID] * 3,
                        (this.model.mesh_normaladr[meshID] + this.model.mesh_normalnum[meshID]) * 3
                    );

                    const normal_buffer = new Float32Array(this.model.mesh_facenum[meshID] * 3 * 3);

                    // UVs
                    let uvs = null;
                    let uv_buffer = null;

                    if (this.model.mesh_texcoordadr[meshID] != -1)
                    {
                        uvs = this.model.mesh_texcoord.subarray(
                            this.model.mesh_texcoordadr[meshID] * 2,
                            (this.model.mesh_texcoordadr[meshID] + this.model.mesh_texcoordnum[meshID]) * 2
                        );

                        uv_buffer = new Float32Array(this.model.mesh_facenum[meshID] * 3 * 2);
                    }

                    const offset = this.model.mesh_faceadr[meshID] * 3;

                    for (let i = 0; i < this.model.mesh_facenum[meshID] * 3; ++i)
                    {
                        const l = this.model.mesh_face[offset + i];

                        vertex_buffer[i * 3] = vertices[l * 3];
                        vertex_buffer[i * 3 + 1] = vertices[l * 3 + 1];
                        vertex_buffer[i * 3 + 2] = vertices[l * 3 + 2];

                        const j = this.model.mesh_facenormal[offset + i];

                        normal_buffer[i * 3] = normals[j * 3];
                        normal_buffer[i * 3 + 1] = normals[j * 3 + 1];
                        normal_buffer[i * 3 + 2] = normals[j * 3 + 2];

                        if (uv_buffer != null)
                        {
                            const k = this.model.mesh_facetexcoord[offset + i];

                            uv_buffer[i * 2] = uvs[k * 2];
                            uv_buffer[i * 2 + 1] = uvs[k * 2 + 1];
                        }
                    }

                    geometry.setAttribute("position", new THREE.BufferAttribute(vertex_buffer, 3));
                    geometry.setAttribute("normal", new THREE.BufferAttribute(normal_buffer, 3));

                    if (uv_buffer != null)
                        geometry.setAttribute("uv", new THREE.BufferAttribute(uv_buffer, 2));

                    this.meshes[meshID] = geometry;
                } else {
                    geometry = this.meshes[meshID];
                }

                this.bodies[b].has_custom_mesh = true;
            }

            // Set the material properties
            let material = defaultMaterial.clone();
            let texture = null;
            let texuniform = false;
            let color = [
                this.model.geom_rgba[(g * 4) + 0],
                this.model.geom_rgba[(g * 4) + 1],
                this.model.geom_rgba[(g * 4) + 2],
                this.model.geom_rgba[(g * 4) + 3]
            ];

            if (this.model.geom_matid[g] != -1) {
                let matId = this.model.geom_matid[g];
                color = [
                    this.model.mat_rgba[(matId * 4) + 0],
                    this.model.mat_rgba[(matId * 4) + 1],
                    this.model.mat_rgba[(matId * 4) + 2],
                    this.model.mat_rgba[(matId * 4) + 3]
                ];

                // Retrieve or construct the texture
                let texId = this.model.mat_texid[matId * mujoco.mjtTextureRole.mjNTEXROLE.value + mujoco.mjtTextureRole.mjTEXROLE_RGB.value];
                if (texId != -1) {
                    if (!(texId in this.textures))
                        texture = this._createTexture(texId);
                    else
                        texture = this.textures[texId];

                    const texrepeat_u = this.model.mat_texrepeat[matId * 2];
                    const texrepeat_v = this.model.mat_texrepeat[matId * 2 + 1];
                    texuniform = (this.model.mat_texuniform[matId] == 1);

                    if ((texrepeat_u != 1.0) || (texrepeat_v != 1.0)) {
                        texture = texture.clone();
                        texture.needsUpdate = true;
                        texture.repeat.x = texrepeat_u;
                        texture.repeat.y = texrepeat_v;
                    }

                    material = new THREE.MeshPhongMaterial({
                        color: new THREE.Color().setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace),
                        transparent: color[3] < 1.0,
                        opacity: color[3],
                        specular: new THREE.Color().setRGB(this.model.mat_specular[matId], this.model.mat_specular[matId], this.model.mat_specular[matId], THREE.SRGBColorSpace),
                        shininess: this.model.mat_shininess[matId],
                        reflectivity: this.model.mat_reflectance[matId],
                        emissive: new THREE.Color().setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace).multiplyScalar(this.model.mat_emission[matId]),
                        map: texture
                    });

                } else if (material.color.r != color[0] ||
                           material.color.g != color[1] ||
                           material.color.b != color[2] ||
                           material.opacity != color[3]) {

                    material = new THREE.MeshPhongMaterial({
                        color: new THREE.Color().setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace),
                        transparent: color[3] < 1.0,
                        opacity: color[3],
                        specular: new THREE.Color().setRGB(this.model.mat_specular[matId], this.model.mat_specular[matId], this.model.mat_specular[matId], THREE.SRGBColorSpace),
                        shininess: this.model.mat_shininess[matId],
                        reflectivity: this.model.mat_reflectance[matId],
                        emissive: new THREE.Color().setRGB(color[0], color[1], color[2]).multiplyScalar(this.model.mat_emission[matId], THREE.SRGBColorSpace),
                    });
                }

            } else if (material.color.r != color[0] ||
                       material.color.g != color[1] ||
                       material.color.b != color[2] ||
                       material.opacity != color[3]) {

                material = new THREE.MeshPhongMaterial({
                    color: new THREE.Color().setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace),
                    transparent: color[3] < 1.0,
                    opacity: color[3],
                });
            }

            // Create the mesh
            let mesh = new THREE.Mesh();
            if (type == mujoco.mjtGeom.mjGEOM_PLANE.value) {
                // mesh = new Reflector(new THREE.PlaneGeometry(100, 100), {
                //     clipBias: 0.003,
                //     texture: texture
                // });

                const infiniteX = (size[0] == 0);
                const infiniteY = (size[1] == 0);
                const spacing = (size[2] == 0 ? 1 : size[2]);

                const width = (infiniteX ? 1 : size[0] * 2.0);
                const height = (infiniteY ? 1 : size[1] * 2.0);

                const widthSegments = (infiniteX ? this.freeCameraSettings.zfar * 2 / spacing : width / spacing);
                const heightSegments = (infiniteY ? this.freeCameraSettings.zfar * 2 / spacing : height / spacing);

                mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, height, widthSegments, heightSegments), material);
                mesh.infiniteX = infiniteX;
                mesh.infiniteY = infiniteY;
                mesh.infinite = infiniteX && infiniteY;
                mesh.texuniform = texuniform;

                if (infiniteX || infiniteY)
                    this.infinitePlanes.push(mesh);

                if (texuniform) {
                    if (!infiniteX)
                        material.map.repeat.x *= size[0];

                    if (!infiniteY)
                        material.map.repeat.y *= size[1];
                }

                if (mesh.infinite && (this.infinitePlane == null))
                    this.infinitePlane = mesh;
            } else {
                mesh = new THREE.Mesh(geometry, material);

                if (texuniform) {
                    material.map.repeat.x *= size[0];
                    material.map.repeat.y *= size[1];
                }
            }

            mesh.castShadow = (g == 0 ? false : true);
            mesh.receiveShadow = true; //(type != 7);
            mesh.bodyId = b;
            this.bodies[b].add(mesh);

            this._getPosition(this.model.geom_pos, g, mesh.position);
            this._getQuaternion(this.model.geom_quat, g, mesh.quaternion);

            if (type == mujoco.mjtGeom.mjGEOM_PLANE.value) {
                if (!mesh.infinite) {
                    const material2 = material.clone();
                    material2.side = THREE.BackSide;
                    material2.transparent = true;
                    material2.opacity = 0.5;

                    const mesh2 = mesh.clone();
                    mesh2.material = material2;

                    this.bodies[b].add(mesh2);
                }
            }

            // Stretch the ellipsoids
            if (type == mujoco.mjtGeom.mjGEOM_ELLIPSOID.value)
                mesh.scale.set(size[0], size[1], size[2]);

            // Change the orientation of some threejs mesh types
            if ((type == mujoco.mjtGeom.mjGEOM_CYLINDER.value) ||
                (type == mujoco.mjtGeom.mjGEOM_CAPSULE.value)) {
                mesh.rotateX(-Math.PI / 2.0);
            }
        }

        // Construct the hierarchy of bodies
        for (let b = 0; b < this.model.nbody; ++b) {
            // Body without geometry, create a three.js group
            if (!this.bodies[b]) {
                this.bodies[b] = new THREE.Group();
                this.bodies[b].name = this.names[b + 1];
                this.bodies[b].bodyId = b;
                this.bodies[b].has_custom_mesh = false;
            }

            const body = this.bodies[b];

            let parent_body = this.model.body_parentid[b];
            if (parent_body == 0)
                this.root.add(body);
            else
                this.bodies[parent_body].add(body);
        }
    }


    _processLights(lightIntensities) {
        const sim = this;

        function _createOrUpdateAmbientLight(color, intensity=1.0) {
            if (sim.ambientLight == null) {
                sim.ambientLight = new THREE.AmbientLight(sim.headlightSettings.ambient, intensity);
                sim.ambientLight.layers.enableAll();
                sim.root.add(sim.ambientLight);
            } else {
                sim.ambientLight.color += color;
            }
        }

        if (this.headlightSettings.active) {
            if ((this.headlightSettings.ambient.r > 0.0) || (this.headlightSettings.ambient.g > 0.0) ||
                (this.headlightSettings.ambient.b > 0.0)) {
                _createOrUpdateAmbientLight(this.headlightSettings.ambient, this.headlightSettings.ambientIntensity);
            }

            if ((this.headlightSettings.diffuse.r > 0.0) || (this.headlightSettings.diffuse.g > 0.0) ||
                (this.headlightSettings.diffuse.b > 0.0)) {
                this.headlight = new THREE.DirectionalLight(this.headlightSettings.diffuse, this.headlightSettings.intensity);
                this.headlight.layers.enableAll();
                this.root.add(this.headlight);
            }
        }

        const dir = new THREE.Vector3();

        for (let l = 0; l < this.model.nlight; ++l) {
            let light = null;

            const intensity = lightIntensities[l];

            if (this.model.light_directional[l])
                light = new THREE.DirectionalLight(0xffffff, intensity != null ? intensity : 3);
            else
                light = new THREE.SpotLight(0xffffff, intensity != null ? intensity : 8);

            light.quaternion.set(0, 0, 0, 1);

            this._getPosition(this.model.light_pos, l, light.position);

            this._getPosition(this.model.light_dir, l, dir);
            dir.add(light.position);

            light.target.position.copy(dir);

            light.color.setRGB(
                this.model.light_diffuse[l * 3],
                this.model.light_diffuse[l * 3 + 1],
                this.model.light_diffuse[l * 3 + 2],
                THREE.SRGBColorSpace
            );

            if (!this.model.light_directional[l]) {
                // light.distance = this.model.light_attenuation[l * 3 + 1];
                // light.decay = this.model.light_attenuation[l * 3 + 1];
                light.penumbra = 0.5;
                light.angle = this.model.light_cutoff[l] * Math.PI / 180.0;
            }

            light.castShadow = this.model.light_castshadow[l];
            if (light.castShadow)
            {
                light.shadow.camera.near = 0.1;
                light.shadow.camera.far = 50;
                light.shadow.mapSize.width = 2048;
                light.shadow.mapSize.height = 2048;
            }

            const b = this.model.light_bodyid[l];
            if (b >= 0)
                this.bodies[b].add(light);
            else
                this.root.add(light);

            this.root.add(light.target);

            light.layers.enableAll();

            this.lights.push(light);

            if ((this.model.light_ambient[l * 3] > 0.0) || (this.model.light_ambient[l * 3 + 1] > 0.0) ||
                (this.model.light_ambient[l * 3 + 2] > 0.0)) {
                _createOrUpdateAmbientLight(
                    new THREE.Color().setRGB(
                        this.model.light_ambient[l * 3],
                        this.model.light_ambient[l * 3 + 1],
                        this.model.light_ambient[l * 3 + 2]),
                        THREE.SRGBColorSpace
                );
            }
        }
    }


    _processSites() {
        for (let s = 0; s < this.model.nsite; ++s) {
            let site = new THREE.Object3D();
            site.site_id = s;

            this._getPosition(this.model.site_pos, s, site.position);
            this._getQuaternion(this.model.site_quat, s, site.quaternion);

            const b = this.model.site_bodyid[s];
            if (b >= 0)
                this.bodies[b].add(site);
            else
                this.root.add(site);

            this.sites[s] = site;
        }
    }


    _createTexture(texId) {
        let width = this.model.tex_width[texId];
        let height = this.model.tex_height[texId];
        let offset = this.model.tex_adr[texId];
        let type = this.model.tex_type[texId];
        let rgbArray = this.model.tex_data;
        let rgbaArray = new Uint8Array(width * height * 4);

        for (let p = 0; p < width * height; p++) {
            rgbaArray[(p * 4) + 0] = rgbArray[offset + ((p * 3) + 0)];
            rgbaArray[(p * 4) + 1] = rgbArray[offset + ((p * 3) + 1)];
            rgbaArray[(p * 4) + 2] = rgbArray[offset + ((p * 3) + 2)];
            rgbaArray[(p * 4) + 3] = 1.0;
        }

        if ((type == mujoco.mjtTexture.mjTEXTURE_SKYBOX.value) && (height == width * 6)) {
            const textures = [];
            for (let i = 0; i < 6; ++i) {
                const size = width * width * 4;

                const texture = new THREE.DataTexture(
                    rgbaArray.subarray(i * size, (i + 1) * size), width, width, THREE.RGBAFormat,
                    THREE.UnsignedByteType
                );

                texture.colorSpace = THREE.SRGBColorSpace;
                texture.flipY = true;
                texture.needsUpdate = true;
                textures.push(texture);
            }

            this.textures[texId] = textures;
            return textures;

        } else {
            const texture = new THREE.DataTexture(rgbaArray, width, height, THREE.RGBAFormat, THREE.UnsignedByteType);
            texture.wrapS = THREE.RepeatWrapping;
            texture.wrapT = THREE.RepeatWrapping;
            texture.needsUpdate = true;
            texture.colorSpace = THREE.SRGBColorSpace;

            this.textures[texId] = texture;
            return texture;
        }
    }


    /** Access the vector at index and store it in the target THREE.Vector3
     * @param {Float32Array|Float64Array} buffer
     * @param {number} index
     * @param {THREE.Vector3} target */
    _getPosition(buffer, index, target) {
        return target.set(
            buffer[(index * 3) + 0],
            buffer[(index * 3) + 1],
            buffer[(index * 3) + 2]
        );
    }


    /**
     * @param {Float32Array|Float64Array} buffer
     * @param {number} index
     * @param {THREE.Vector3} src */
    _setPosition(buffer, index, src) {
        buffer[(index * 3) + 0] = src.x;
        buffer[(index * 3) + 1] = src.y;
        buffer[(index * 3) + 2] = src.z;
    }


    /** Access the quaternion at index and store it in the target THREE.Quaternion
     * @param {Float32Array|Float64Array} buffer
     * @param {number} index
     * @param {THREE.Quaternion} target */
    _getQuaternion(buffer, index, target) {
        return target.set(
            buffer[(index * 4) + 1],
            buffer[(index * 4) + 2],
            buffer[(index * 4) + 3],
            buffer[(index * 4) + 0]
        );
    }


    _getMatrix(buffer, index, target) {
        return target.set(
            buffer[(index * 9) + 0],
            buffer[(index * 9) + 1],
            buffer[(index * 9) + 2],
            buffer[(index * 9) + 3],
            buffer[(index * 9) + 4],
            buffer[(index * 9) + 5],
            buffer[(index * 9) + 6],
            buffer[(index * 9) + 7],
            buffer[(index * 9) + 8]
        );
    }


    _computeStatistics(statistics) {
        // This method is a port of the corresponding one in MuJoCo
        this.statistics = {
            extent: 2.0,
            center: new THREE.Vector3(),
            meansize: 0.0,
            meanmass: 0.0,
            meaninertia: 0.0,
        };

        var bbox = new THREE.Box3();
        var point = new THREE.Vector3();

        // Compute bounding box of bodies, joint centers, geoms and sites
        for (let i = 1; i < this.model.nbody; ++i) {
            point.set(this.simulation.xpos[3*i], this.simulation.xpos[3*i+1], this.simulation.xpos[3*i+2]);
            bbox.expandByPoint(point);

            point.set(this.simulation.xipos[3*i], this.simulation.xipos[3*i+1], this.simulation.xipos[3*i+2]);
            bbox.expandByPoint(point);
        }

        for (let i = 0; i < this.model.njnt; ++i) {
            point.set(this.simulation.xanchor[3*i], this.simulation.xanchor[3*i+1], this.simulation.xanchor[3*i+2]);
            bbox.expandByPoint(point);
        }

        for (let i = 0; i < this.model.nsite; ++i) {
            point.set(this.simulation.site_xpos[3*i], this.simulation.site_xpos[3*i+1], this.simulation.site_xpos[3*i+2]);
            bbox.expandByPoint(point);
        }

        for (let i = 0; i < this.model.ngeom; ++i) {
            // set rbound: regular geom rbound, or 0.1 of plane or hfield max size
            let rbound = 0.0;

            if (this.model.geom_rbound[i] > 0.0) {
                rbound = this.model.geom_rbound[i];
            } else if (this.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE.value) {
                // finite in at least one direction
                if ((this.model.geom_size[3*i] > 0.0) || (this.model.geom_size[3*i+1] > 0.0)) {
                    rbound = Math.max(this.model.geom_size[3*i], this.model.geom_size[3*i+1]) * 0.1;
                }

                // infinite in both directions
                else {
                    rbound = 1.0;
                }
            } else if (this.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_HFIELD.value) {
                const j = this.model.geom_dataid[i];
                rbound = Math.max(this.model.hfield_size[4*j],
                                  this.model.hfield_size[4*j+1],
                                  this.model.hfield_size[4*j+2],
                                  this.model.hfield_size[4*j+3]
                                 ) * 0.1;
            }

            point.set(this.simulation.geom_xpos[3*i] + rbound, this.simulation.geom_xpos[3*i+1] + rbound, this.simulation.geom_xpos[3*i+2] + rbound);
            bbox.expandByPoint(point);

            point.set(this.simulation.geom_xpos[3*i] - rbound, this.simulation.geom_xpos[3*i+1] - rbound, this.simulation.geom_xpos[3*i+2] - rbound);
            bbox.expandByPoint(point);
        }

        // Compute center
        bbox.getCenter(this.statistics.center);

        // compute bounding box size
        if (bbox.max.x > bbox.min.x) {
            const size = new THREE.Vector3();
            bbox.getSize(size);
            this.statistics.extent = Math.max(1e-5, size.x, size.y, size.z);
        }

        // set body size to max com-joint distance
        const body = new Array(this.model.nbody);
        for (let i = 0; i < this.model.nbody; ++i)
            body[i] = 0.0;

        var point2 = new THREE.Vector3();

        for (let i = 0; i < this.model.njnt; ++i) {
            // handle this body
            let id = this.model.jnt_bodyid[i];
            point.set(this.simulation.xipos[3*id], this.simulation.xipos[3*id+1], this.simulation.xipos[3*id+2]);
            point2.set(this.simulation.xanchor[3*i], this.simulation.xanchor[3*i+1], this.simulation.xanchor[3*i+2]);

            body[id] = Math.max(body[id], point.distanceTo(point2));

            // handle parent body
            id = this.model.body_parentid[id];
            point.set(this.simulation.xipos[3*id], this.simulation.xipos[3*id+1], this.simulation.xipos[3*id+2]);
            point2.set(this.simulation.xanchor[3*i], this.simulation.xanchor[3*i+1], this.simulation.xanchor[3*i+2]);

            body[id] = Math.max(body[id], point.distanceTo(point2));
        }
        body[0] = 0.0;

        // set body size to max of old value, and geom rbound + com-geom dist
        for (let i = 1; i < this.model.nbody; ++i) {
            for (let id = this.model.body_geomadr[i]; id < this.model.body_geomadr[i] + this.model.body_geomnum[i]; ++id) {
                if (this.model.geom_rbound[id] > 0) {
                    point.set(this.simulation.xipos[3*i], this.simulation.xipos[3*i+1], this.simulation.xipos[3*i+2]);
                    point2.set(this.simulation.geom_xpos[3*id], this.simulation.geom_xpos[3*id+1], this.simulation.geom_xpos[3*id+2]);
                    body[i] = Math.max(body[i], this.model.geom_rbound[id] + point.distanceTo(point2));
                }
            }
        }

        // compute meansize, make sure all sizes are above min
        if (this.model.nbody > 1) {
            this.statistics.meansize = 0.0;
            for (let i = 1; i < this.model.nbody; ++i) {
                body[i] = Math.max(body[i], 1e-5);
                this.statistics.meansize += body[i] / (this.model.nbody - 1);
            }
        }

        // fix extent if too small compared to meanbody
        this.statistics.extent = Math.max(this.statistics.extent, 2 * this.statistics.meansize);

        // compute meanmass
        if (this.model.nbody > 1) {
            this.statistics.meanmass = 0.0;
            for (let i = 1; i < this.model.nbody; ++i)
                this.statistics.meanmass += this.model.body_mass[i];
            this.statistics.meanmass /= (this.model.nbody - 1);
        }

        // compute meaninertia
        if (this.model.nv > 0) {
            this.statistics.meaninertia = 0.0;
            for (let i = 0; i < this.model.nv; ++i)
                this.statistics.meaninertia += this.simulation.qM[this.model.dof_Madr[i]];
            this.statistics.meaninertia /= this.model.nv;
        }

        // Override with the values found in the XML file
        this.statistics.extent = statistics.extent || this.statistics.extent;
        this.statistics.center = statistics.center || this.statistics.center;
        this.statistics.meansize = statistics.meansize || this.statistics.meansize;
        this.statistics.meanmass = statistics.meanmass || this.statistics.meanmass;
        this.statistics.meaninertia = statistics.meaninertia || this.statistics.meaninertia;
    }
}



function loadXmlFile(filename) {
    try {
        const stat = mujoco.FS.stat(filename);
    } catch (ex) {
        return null;
    }

    const textDecoder = new TextDecoder("utf-8");
    const data = textDecoder.decode(mujoco.FS.readFile(filename));

    const parser = new DOMParser();
    return parser.parseFromString(data, "text/xml");
}



function getFreeCameraSettings(xmlDoc) {
    const settings = {
        fovy: 45.0,
        azimuth: 90.0,
        elevation: -45.0,
        znear: 0.01,
        zfar: 50,
    };

    const xmlVisual = getFirstElementByTagName(xmlDoc, "visual");
    if (xmlVisual == null)
        return settings;

    const xmlGlobal = getFirstElementByTagName(xmlVisual, "global");
    if (xmlGlobal != null) {
        let value = xmlGlobal.getAttribute("fovy");
        if (value != null)
            settings.fovy = Number(value);

        value = xmlGlobal.getAttribute("azimuth");
        if (value != null)
            settings.azimuth = Number(value);

        value = xmlGlobal.getAttribute("elevation");
        if (value != null)
            settings.elevation = Number(value);
    }

    const xmlMap = getFirstElementByTagName(xmlVisual, "map");
    if (xmlMap != null) {
        let value = xmlMap.getAttribute("znear");
        if (value != null)
            settings.znear = Number(value);

        value = xmlMap.getAttribute("zfar");
        if (value != null)
            settings.zfar = Number(value);
    }

    return settings;
}


function getStatistics(xmlDoc) {
    const statistics = {
        extent: null,
        center: null,
        meansize: null,
        meanmass: null,
        meaninertia: null,
    };

    const xmlStatistic = getFirstElementByTagName(xmlDoc, "statistic");
    if (xmlStatistic == null)
        return statistics;

    let value = xmlStatistic.getAttribute("extent");
    if (value != null)
        statistics.extent = Number(value);

    value = xmlStatistic.getAttribute("center");
    if (value != null) {
        const v = value.split(" ");
        statistics.center = new THREE.Vector3(Number(v[0]), Number(v[1]), Number(v[2]));
    }

    value = xmlStatistic.getAttribute("meansize");
    if (value != null)
        statistics.meansize = Number(value);

    value = xmlStatistic.getAttribute("meanmass");
    if (value != null)
        statistics.meanmass = Number(value);

    value = xmlStatistic.getAttribute("meaninertia");
    if (value != null)
        statistics.meaninertia = Number(value);

    return statistics;
}


function getFogSettings(xmlDoc) {
    const settings = {
        fogEnabled: false,
        fog: new THREE.Color(0, 0, 0),
        fogStart: 3,
        fogEnd: 10,

        hazeEnabled: false,
        haze: new THREE.Color(1, 1, 1),
        hazeProportion: 0.3,
    };

    const xmlVisual = getFirstElementByTagName(xmlDoc, "visual");
    if (xmlVisual == null)
        return settings;

    const xmlRgba = getFirstElementByTagName(xmlVisual, "rgba");
    if (xmlRgba != null) {
        let value = xmlRgba.getAttribute("fog");
        if (value != null) {
            const v = value.split(" ");
            settings.fog.setRGB(Number(v[0]), Number(v[1]), Number(v[2]), THREE.SRGBColorSpace);
            settings.fogEnabled = true;
        }

        value = xmlRgba.getAttribute("haze");
        if (value != null) {
            const v = value.split(" ");
            settings.haze.setRGB(Number(v[0]), Number(v[1]), Number(v[2]), THREE.SRGBColorSpace);
            settings.hazeEnabled = true;
        }
    }

    const xmlMap = getFirstElementByTagName(xmlVisual, "map");
    if (xmlMap != null) {
        let value = xmlMap.getAttribute("fogstart");
        if (value != null) {
            settings.fogStart = Number(value);
            settings.fogEnabled = true;
        }

        value = xmlMap.getAttribute("fogend");
        if (value != null) {
            settings.fogEnd = Number(value);
            settings.fogEnabled = true;
        }

        value = xmlMap.getAttribute("haze");
        if (value != null) {
            settings.hazeProportion = Number(value);
            settings.hazeEnabled = true;
        }
    }

    return settings;
}


function getHeadlightSettings(xmlDoc, filename) {
    const settings = {
        ambient: new THREE.Color().setRGB(0.1, 0.1, 0.1, THREE.SRGBColorSpace),
        diffuse: new THREE.Color().setRGB(0.4, 0.4, 0.4, THREE.SRGBColorSpace),
        intensity: 3,
        ambientIntensity : 1,
        active: true,
    };

    const xmlVisual = getFirstElementByTagName(xmlDoc, "visual");
    if (xmlVisual == null)
        return settings;

    let modified = false;

    const xmlHeadlight = getFirstElementByTagName(xmlVisual, "headlight");
    if (xmlHeadlight != null) {
        let value = xmlHeadlight.getAttribute("ambient");
        if (value != null) {
            const v = value.split(" ");
            settings.ambient.setRGB(Number(v[0]), Number(v[1]), Number(v[2]), THREE.SRGBColorSpace);
        }

        value = xmlHeadlight.getAttribute("diffuse");
        if (value != null) {
            const v = value.split(" ");
            settings.diffuse.setRGB(Number(v[0]), Number(v[1]), Number(v[2]), THREE.SRGBColorSpace);
        }

        value = xmlHeadlight.getAttribute("active");
        if (value != null)
            settings.active = (value == "1");

        value = xmlHeadlight.getAttribute("intensity");
        if (value != null) {
            settings.intensity = Number(value);
            xmlHeadlight.removeAttribute("intensity");
            modified = true;
        }

        value = xmlHeadlight.getAttribute("ambientIntensity");
        if (value != null) {
            settings.ambientIntensity = Number(value);
            xmlHeadlight.removeAttribute("ambientIntensity");
            modified = true;
        }
    }

    if (modified) {
        const serializer = new XMLSerializer();
        mujoco.FS.writeFile(filename, serializer.serializeToString(xmlDoc));
    }

    return settings;
}


function getLightIntensities(xmlDoc, filename=null) {
    const intensities = [];

    const xmlWorldBody = getFirstElementByTagName(xmlDoc, "worldbody");
    if (xmlWorldBody == null)
        return intensities;

    const xmlLights = xmlWorldBody.getElementsByTagName("light");
    if (xmlLights.length == 0)
        return intensities;

    let modified = false;

    for (let i = 0; i < xmlLights.length; ++i) {
        const xmlLight = xmlLights[i];

        let value = xmlLight.getAttribute("intensity");
        if (value != null) {
            intensities.push(Number(value));
            xmlLight.removeAttribute("intensity");
            modified = true;
        } else {
            intensities.push(null);
        }
    }

    if (modified && (filename != null)) {
        const serializer = new XMLSerializer();
        mujoco.FS.writeFile(filename, serializer.serializeToString(xmlDoc));
    }

    return intensities;
}


function generateRobot(xmlDoc, filename, robotBuilder) {
    // Add an include element in the XML document
    const xmlInclude = xmlDoc.createElement("include");
    xmlInclude.setAttribute("file", "generated/" + robotBuilder.name + ".xml");
    xmlInclude.setAttribute("pos", robotBuilder.position.join(" "));
    xmlInclude.setAttribute("quat", robotBuilder.quaternion.join(" "));

    if (robotBuilder.prefix != null)
        xmlInclude.setAttribute("prefix", robotBuilder.prefix);

    const xmlIncludes = xmlDoc.getElementsByTagName("include");
    if (xmlIncludes.length == 0)
        xmlDoc.children[0].prepend(xmlInclude);
    else
        xmlIncludes[xmlIncludes.length - 1].after(xmlInclude);

    // Generate the XML file declaring the robot
    const xmlRobotDoc = robotBuilder.generateXMLDocument();

    const offset = filename.lastIndexOf('/');
    const folder = filename.substring(0, offset + 1);
    const path = folder + "/generated";

    try {
        const stat = mujoco.FS.stat(path);
    } catch (ex) {
        mujoco.FS.mkdir(path);
    }

    const serializer = new XMLSerializer();
    mujoco.FS.writeFile(
        path + "/" + robotBuilder.name + ".xml",
        serializer.serializeToString(xmlRobotDoc)
    );
}


function preprocessIncludedFiles(xmlDoc, filename, modified=false) {
    const offset = filename.lastIndexOf('/');
    const folder = filename.substring(0, offset + 1);
    const [subfolder, _] = filename.substring(offset + 1).split(".");

    const serializer = new XMLSerializer();

    const knownFiles = [];

    let intensities = [];

    // Search for include directives with a prefix
    const xmlIncludes = xmlDoc.getElementsByTagName("include");
    for (let xmlInclude of xmlIncludes) {
        let includedFile = xmlInclude.getAttribute("file");

        const known = (knownFiles.indexOf(includedFile) != -1);
        if (!known)
            knownFiles.push(includedFile);

        const prefix = xmlInclude.getAttribute("prefix");
        const pos = xmlInclude.getAttribute("pos");
        const quat = xmlInclude.getAttribute("quat");
        const ghost = xmlInclude.getAttribute("ghost");

        const [xmlContent, intensities2] = preprocessIncludedFile(folder + includedFile, prefix, pos, quat, ghost, known);

        if (intensities2.length > 0)
            intensities = intensities.concat(intensities2);

        if ((prefix != null) || (pos != null) || (quat != null) || (ghost != null) || (intensities2.length > 0)) {
            const offset = includedFile.lastIndexOf('/');
            const path = subfolder + "/" + includedFile.substring(0, offset + 1);
            includedFile = path + (prefix != null ? prefix : "") + includedFile.substring(offset + 1);

            mkdir(folder + path);

            mujoco.FS.writeFile(folder + includedFile, serializer.serializeToString(xmlContent));
            xmlInclude.setAttribute("file", includedFile);
            xmlInclude.removeAttribute("prefix");
            xmlInclude.removeAttribute("pos");
            xmlInclude.removeAttribute("quat");
            xmlInclude.removeAttribute("ghost");

            modified = true;
        }
    }

    if (modified)
        mujoco.FS.writeFile(filename, serializer.serializeToString(xmlDoc));

    return intensities;
}


function getFirstElementByTagName(xmlParent, name) {
    const xmlElements = xmlParent.getElementsByTagName(name);
    if (xmlElements.length > 0)
        return xmlElements[0];

    return null;
}


function preprocessIncludedFile(filename, prefix, pos, quat, ghost, removeCommons=false) {
    const xmlDoc = loadXmlFile(filename);
    if (xmlDoc == null) {
        console.error('Missing file: ' + filename);
        return;
    }

    // Retrieve the model name
    const xmlRoot = getFirstElementByTagName(xmlDoc, "mujoco");

    // Process all prefix-related changes
    if (prefix != null) {
        const modelName = xmlRoot.getAttribute("model").replaceAll(' ' , '_');

        if (!removeCommons) {
            // Process defaults
            const xmlDefaults = getFirstElementByTagName(xmlRoot, "default");
            if (xmlDefaults != null)
                changeDefaultClassNames(xmlDefaults, modelName + '_');

            // Process assets
            const xmlAssets = getFirstElementByTagName(xmlRoot, "asset");
            if (xmlAssets != null)
                changeAssetNames(xmlAssets, modelName + '_');
        } else {
            const xmlDefaults = getFirstElementByTagName(xmlRoot, "default");
            if (xmlDefaults != null)
                xmlRoot.removeChild(xmlDefaults);

            const xmlAssets = getFirstElementByTagName(xmlRoot, "asset");
            if (xmlAssets != null)
                xmlRoot.removeChild(xmlAssets);
        }

        const elements = ["worldbody", "tendon", "equality", "actuator", "contact", "sensor"];
        for (let name of elements) {
            const xmlElement = getFirstElementByTagName(xmlRoot, name);
            if (xmlElement != null)
                changeNames(xmlElement, prefix, modelName);
        }
    }

    // Process all transforms-related changes
    const xmlWorldBody = getFirstElementByTagName(xmlRoot, "worldbody");
    if (xmlWorldBody != null) {
        if (pos != null) {
            for (let xmlChild of xmlWorldBody.children) {
                const v = pos.split(" ");
                const finalPos = new THREE.Vector3(Number(v[0]), Number(v[1]), Number(v[2]));

                const origPos = xmlChild.getAttribute("pos");
                if (origPos != null) {
                    const v = origPos.split(" ");
                    finalPos.x += Number(v[0]);
                    finalPos.y += Number(v[1]);
                    finalPos.z += Number(v[2]);
                }

                xmlChild.setAttribute("pos", "" + finalPos.x + " " + finalPos.y + " " + finalPos.z);
            }
        }

        if (quat != null) {
            let useDegrees = false;
            let eulerSeq = "XYZ";

            const xmlCompiler = getFirstElementByTagName(xmlRoot, "compiler");
            if (xmlCompiler != null) {
                let value = xmlCompiler.getAttribute("angle");
                useDegrees = (value == "degree");

                value = xmlCompiler.getAttribute("eulerseq");
                if (value != null)
                    eulerSeq = value.toUpperCase();
            }

            for (let xmlChild of xmlWorldBody.children) {
                const v = quat.split(" ");
                const finalQuat = new THREE.Quaternion(Number(v[1]), Number(v[2]), Number(v[3]), Number(v[0]));

                const origQuat = xmlChild.getAttribute("quat");
                const origAxisAngle = xmlChild.getAttribute("axisangle");
                const origEuler = xmlChild.getAttribute("euler");
                const origXYAxes = xmlChild.getAttribute("xyaxes");
                const origZAxis = xmlChild.getAttribute("zaxis");

                if (origQuat != null) {
                    const v = origQuat.split(" ");
                    const quat = new THREE.Quaternion(Number(v[1]), Number(v[2]), Number(v[3]), Number(v[0]));
                    finalQuat.multiply(quat);

                } else if (origAxisAngle != null) {
                    const v = origAxisAngle.split(" ");
                    const x = Number(v[0]);
                    const y = Number(v[1]);
                    const z = Number(v[2]);
                    let a = Number(v[3]);

                    if (useDegrees)
                        a = a * Math.PI / 180.0;

                    if (a != 0.0) {
                        const s = Math.sin(a * 0.5);
                        const quat = new THREE.Quaternion(x * s, y * s, z * s, Math.cos(a * 0.5));
                        finalQuat.multiply(quat);
                    }

                    xmlChild.removeAttribute("axisangle");

                } else if (origEuler != null) {
                    const v = origEuler.split(" ");
                    const x = Number(v[0]);
                    const y = Number(v[1]);
                    const z = Number(v[2]);

                    const euler = new THREE.Euler(x, y, z, eulerSeq);

                    const quat = new THREE.Quaternion();
                    quat.setFromEuler(euler);

                    finalQuat.multiply(quat);

                    xmlChild.removeAttribute("euler");

                } else if (origXYAxes != null) {
                    const v = origXYAxes.split(" ");
                    const x = new THREE.Vector3(Number(v[0]), Number(v[1]), Number(v[2]));
                    const y = new THREE.Vector3(Number(v[3]), Number(v[4]), Number(v[5]));

                    x.normalize();
                    y.normalize();

                    const z = new THREE.Vector3();
                    z.crossVectors(x, y);

                    const matrix = new THREE.Matrix4();
                    matrix.makeBasis(x, y, z);

                    const quat = new THREE.Quaternion();
                    quat.setFromRotationMatrix(matrix);

                    quat.set(quat.x, quat.y, quat.z, quat.w);

                    finalQuat.multiply(quat);

                    xmlChild.removeAttribute("xyaxes");

                } else if (origZAxis != null) {
                    const v = origZAxis.split(" ");
                    const to = new THREE.Vector3(Number(v[0]), Number(v[1]), Number(v[2]));
                    const from = new THREE.Vector3(0, 0, 1);

                    to.normalize();

                    const quat = new THREE.Quaternion();
                    quat.setFromUnitVectors(from, to);

                    quat.set(quat.x, quat.y, quat.z, quat.w);

                    finalQuat.multiply(quat);

                    xmlChild.removeAttribute("zaxis");
                }

                xmlChild.setAttribute("quat", "" + -finalQuat.w + " " + -finalQuat.x + " " + -finalQuat.y + " " + finalQuat.z);
            }
        }
    }

    // Process all ghost-related changes
    if ((xmlWorldBody != null) && (ghost == "true"))
    {
        const xmlGeoms = Array.from(xmlWorldBody.getElementsByTagName("geom"));
        for (let xmlGeom of xmlGeoms) {
            if (xmlGeom.getAttribute("class").search("collision") >= 0)
                xmlGeom.parentElement.removeChild(xmlGeom);
        }
    }

    const intensities = getLightIntensities(xmlDoc);

    return [xmlDoc, intensities];
}


function addPrefix(xmlElement, attr, prefix) {
    if (xmlElement.hasAttribute(attr))
        xmlElement.setAttribute(attr, prefix + xmlElement.getAttribute(attr));
}


function changeNames(xmlElement, prefix, modelName) {
    addPrefix(xmlElement, "name", prefix);
    addPrefix(xmlElement, "joint", prefix);
    addPrefix(xmlElement, "joint1", prefix);
    addPrefix(xmlElement, "joint2", prefix);
    addPrefix(xmlElement, "tendon", prefix);
    addPrefix(xmlElement, "geom1", prefix);
    addPrefix(xmlElement, "geom2", prefix);
    addPrefix(xmlElement, "site", prefix);
    addPrefix(xmlElement, "target", prefix);
    addPrefix(xmlElement, "prefix", prefix);

    addPrefix(xmlElement, "childclass", modelName + "_");
    addPrefix(xmlElement, "class", modelName + "_");
    addPrefix(xmlElement, "mesh", modelName + "_");
    addPrefix(xmlElement, "material", modelName + "_");
    addPrefix(xmlElement, "hfield", modelName + "_");

    for (let xmlChild of xmlElement.children)
        changeNames(xmlChild, prefix, modelName);
}


function changeDefaultClassNames(xmlElement, prefix) {
    addPrefix(xmlElement, "class", prefix);

    for (let xmlChild of xmlElement.children)
        changeDefaultClassNames(xmlChild, prefix);
}


function changeAssetNames(xmlElement, prefix) {
    addPrefix(xmlElement, "name", prefix);
    addPrefix(xmlElement, "class", prefix);
    addPrefix(xmlElement, "texture", prefix);
    addPrefix(xmlElement, "material", prefix);
    addPrefix(xmlElement, "body", prefix);

    if ((xmlElement.tagName == "texture") || (xmlElement.tagName == "hfield") ||
        (xmlElement.tagName == "mesh") || (xmlElement.tagName == "skin")) {
        if (!xmlElement.hasAttribute("name")) {
            const file = xmlElement.getAttribute("file");
            if (file != null) {
                const offset = file.lastIndexOf('/');
                const offset2 = file.lastIndexOf('.');
                xmlElement.setAttribute("name", prefix + file.substring(offset + 1, offset2));
            }
        }
    }

    for (let xmlChild of xmlElement.children)
        changeAssetNames(xmlChild, prefix);
}

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Download some files and store them in Mujoco's filesystem

Parameters:
    dstFolder (string): The destination folder in Mujoco's filesystem
    srcFolderUrl (string): The URL of the folder in which all the files are located
    filenames ([string]): List of the filenames
*/
async function downloadFiles(dstFolder, srcFolderUrl, filenames) {
    const FS = mujoco.FS;

    if (dstFolder[0] != '/') {
        console.error('Destination folders must be absolute paths starting with /');
        return;
    }

    if (dstFolder.length > 1) {
        if (dstFolder[dstFolder.length-1] == '/')
            dstFolder = dstFolder.substr(0, dstFolder.length-1);

        const parts = dstFolder.substring(1).split('/');

        let path = '';
        for (let i = 0; i < parts.length; ++i) {
            path += '/' + parts[i];

            try {
                const stat = FS.stat(path);
            } catch (ex) {
                FS.mkdir(path);
            }
        }
    }

    if (srcFolderUrl[srcFolderUrl.length-1] != '/')
        srcFolderUrl += '/';

    for (let i = 0; i < filenames.length; ++i) {
        const filename = filenames[i];
        const data = await fetch(srcFolderUrl + filename);

        const contentType = data.headers.get("content-type");
        if ((contentType == 'application/xml') || (contentType == 'text/plain')) {
            mujoco.FS.writeFile(dstFolder + '/' + filename, await data.text());
        } else {
            mujoco.FS.writeFile(dstFolder + '/' + filename, new Uint8Array(await data.arrayBuffer()));
        }
    }
}



/* Download a scene file

The scenes are stored in Mujoco's filesystem at '/scenes'
*/
async function downloadScene(url, destFolder='/scenes') {
    const offset = url.lastIndexOf('/');
    await downloadFiles(destFolder, url.substring(0, offset), [ url.substring(offset + 1) ]);
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Provides controls to translate/rotate an object, either in world or local coordinates.

Buttons are displayed when a transformation is in progress, to switch between
translation/rotation and world/local coordinates.
*/
class TransformControlsManager {

    /* Construct the manager.

    Parameters:
        domElement (element): The DOM element used by the 3D viewer
        rendererElement (element): The DOM element used by the renderer
        camera (Camera): The camera used to render the scene
        scene (Scene): The scene containing the objects to manipulate
    */
    constructor(toolbar, rendererElement, camera, scene) {
        this.transformControls = new TransformControls(camera, rendererElement);
        this.transformControls.manager = this;

        // function _changeLayer(obj) {
        //     if (obj.type == "Mesh") {
        //         obj.layers.disableAll();
        //         obj.layers.enable(2);
        //     };
        //
        //     for (const child of obj.children)
        //         _changeLayer(child);
        // }
        //
        // _changeLayer(this.transformControls.getHelper());
        //
        // console.log(this.transformControls.getHelper());

        scene.add(this.transformControls.getHelper());

        this.btnTranslation = null;
        this.btnRotation = null;
        this.btnScaling = null;
        this.btnWorld = null;
        this.btnLocal = null;

        this.enabled = true;
        this.used = false;

        this.toolbarSection = toolbar.addSection();
        this._createButtons(this.toolbarSection);

        this.enable(false);

        this.listener = null;

        this.transformControls.addEventListener('mouseDown', evt => this.used = true);
    }


    /* Sets up a function that will be called whenever the specified event happens
    */
    addEventListener(name, fct) {
        this.transformControls.addEventListener(name, fct);
    }


    /* Enables/disables the controls
    */
    enable(enabled, withScaling=false) {
        this.enabled = enabled && (this.transformControls.object != null);
        this.used = false;

        if (this.enabled) {
            this.toolbarSection.style.display = 'inline-block';
            this.transformControls.visible = true;

            if (withScaling) {
                this.btnScaling.style.display = 'inline-flex';
            }
            else {
                if (this.btnScaling.classList.contains('activated'))
                    this._onTranslationButtonClicked();

                this.btnScaling.style.display = 'none';
            }
        } else {
            this.toolbarSection.style.display = 'none';
            this.transformControls.visible = false;
        }
    }


    /* Indicates whether or not the controls are enabled
    */
    isEnabled() {
        return this.enabled;
    }


    /* Indicates whether or not dragging is currently performed
    */
    isDragging() {
        return this.transformControls.dragging;
    }


    /* Indicates whether or not the controls were just used
    */
    wasUsed() {
        const result = this.used;
        this.used = false;
        return result;
    }


    /* Sets the 3D object that should be transformed and ensures the controls UI is visible.

    Parameters:
        object (Object3D): The 3D object that should be transformed
    */
    attach(object, withScaling=false, listener=null) {
        if (this.listener != null)
        {
            this.transformControls.removeEventListener('change', this._onTransformsChanged);
            this.transformControls.removeEventListener('mouseUp', this._onMouseUp);
            this.listener = null;
        }

        this.transformControls.attach(object);
        this.enable(true, withScaling);
        this.used = true;

        if (listener != null)
        {
            this.listener = listener;
            this.transformControls.addEventListener('change', this._onTransformsChanged);
            this.transformControls.addEventListener('mouseUp', this._onMouseUp);
        }
    }


    /* Removes the current 3D object from the controls and makes the helper UI invisible.
    */
    detach() {
        this.transformControls.detach();
        this.enable(false);
        this.used = false;

        if (this.listener != null)
        {
            this.transformControls.removeEventListener('change', this._onTransformsChanged);
            this.transformControls.removeEventListener('mouseUp', this._onMouseUp);
            this.listener = null;
        }
    }


    getAttachedObject() {
        return this.transformControls.object;
    }


    _createButtons(section) {
        this.btnTranslation = document.createElement('button');
        this.btnTranslation.innerText = 'Translation';
        this.btnTranslation.className = 'left activated';
        section.appendChild(this.btnTranslation);

        this.btnScaling = document.createElement('button');
        this.btnScaling.innerText = 'Scaling';
        section.appendChild(this.btnScaling);

        this.btnRotation = document.createElement('button');
        this.btnRotation.innerText = 'Rotation';
        this.btnRotation.className = 'right';
        section.appendChild(this.btnRotation);

        this.btnWorld = document.createElement('button');
        this.btnWorld.innerText = 'World';
        this.btnWorld.className = 'left spaced activated';
        section.appendChild(this.btnWorld);

        this.btnLocal = document.createElement('button');
        this.btnLocal.innerText = 'Local';
        this.btnLocal.className = 'right';
        section.appendChild(this.btnLocal);

        this.btnTranslation.addEventListener('click', evt => this._onTranslationButtonClicked(evt));
        this.btnScaling.addEventListener('click', evt => this._onScalingButtonClicked(evt));
        this.btnRotation.addEventListener('click', evt => this._onRotationButtonClicked(evt));
        this.btnWorld.addEventListener('click', evt => this._onWorldButtonClicked(evt));
        this.btnLocal.addEventListener('click', evt => this._onLocalButtonClicked(evt));
    }


    _onTranslationButtonClicked(event) {
        if (this.transformControls.mode == 'translate')
            return;

        this.btnRotation.classList.remove('activated');
        this.btnScaling.classList.remove('activated');
        this.btnTranslation.classList.add('activated');
        this.transformControls.setMode('translate');

        this.btnLocal.disabled = false;
        this.btnWorld.disabled = false;

        this.btnLocal.classList.remove('disabled');
        this.btnWorld.classList.remove('disabled');
    }


    _onScalingButtonClicked(event) {
        if (this.transformControls.mode == 'scale')
            return;

        this.btnTranslation.classList.remove('activated');
        this.btnRotation.classList.remove('activated');
        this.btnScaling.classList.add('activated');
        this.transformControls.setMode('scale');

        this.btnLocal.disabled = true;
        this.btnWorld.disabled = true;

        this.btnLocal.classList.add('disabled');
        this.btnWorld.classList.add('disabled');
    }


    _onRotationButtonClicked(event) {
        if (this.transformControls.mode == 'rotate')
            return;

        this.btnTranslation.classList.remove('activated');
        this.btnScaling.classList.remove('activated');
        this.btnRotation.classList.add('activated');
        this.transformControls.setMode('rotate');

        this.btnLocal.disabled = false;
        this.btnWorld.disabled = false;

        this.btnLocal.classList.remove('disabled');
        this.btnWorld.classList.remove('disabled');
    }


    _onWorldButtonClicked(event) {
        if (this.transformControls.space == 'world')
            return;

        this.btnLocal.classList.remove('activated');
        this.btnWorld.classList.add('activated');
        this.transformControls.setSpace('world');
    }


    _onLocalButtonClicked(event) {
        if (this.transformControls.space == 'local')
            return;

        this.btnWorld.classList.remove('activated');
        this.btnLocal.classList.add('activated');
        this.transformControls.setSpace('local');
    }


    _onTransformsChanged(evt) {
        if ((this.object != null) && this.dragging)
            this.manager.listener(this.object, true);
    }

    _onMouseUp(evt) {
        if ((this.object != null) && this.dragging)
            this.manager.listener(this.object, false);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



class PlanarIKControls {

    constructor() {
        this.robot = null;
        this.offset = null;
        this.kinematicChain = null;

        this.plane = new THREE.Mesh(
            new THREE.PlaneGeometry(100000, 100000, 2, 2),
            new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide})
        );
    }


    setup(robot, offset, kinematicChain, startPosition, planeDirection) {
        this.robot = robot;
        this.offset = offset;
        this.kinematicChain = kinematicChain;

        this.plane.position.copy(startPosition); 

        this.plane.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), planeDirection);
        this.plane.updateMatrixWorld(true);
    }


    process(raycaster) {
        let intersects = raycaster.intersectObject(this.plane, false);

        if (intersects.length > 0) {
            const mu = intersects[0].point;

            this.kinematicChain.ik(
                [mu.x, mu.y, mu.z],
                null,
                [this.offset.x, this.offset.y, this.offset.z]
            );
        }
    }

}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Class in charge of the rendering of a logmap distance on a sphere.

It works by first rendering the sphere, the plane and the various points and lines in a
render target. Then the render target's texture is used in a sprite located in the upper
left corner, in a scene using an orthographic camera.

It is expected that the caller doesn't clear the color buffer (but only its depth buffer),
but render its content on top of it (without any background), after calling the 'render()'
method of the logmap object.
*/
class Logmap {

    /* Constructs the logmap visualiser

    Parameters:
        domElement (element): The DOM element used by the 3D viewer
        size (int): Size of the sphere (in pixels, approximately. Default: 1/10 of the
                    width of the DOM element at creation time)
    */
    constructor(domElement, robot, target, size=null, position='left') {
        this.domElement = domElement;
        this.robot = robot;
        this.target = target;

        this.scene = null;
        this.orthoScene = null;

        this.camera = null;
        this.orthoCamera = null;

        this.render_target = null;

        this.size = (size || Math.round(this.domElement.clientWidth * 0.1)) * 5;
        this.textureSize = this.size * window.devicePixelRatio;

        this.position = position;

        this.sphere = null;
        this.destPoint = null;
        this.destPointCtrl = null;
        this.srcPoint = null;
        this.srcPointCtrl = null;
        this.projectedPoint = null;
        this.line = null;
        this.plane = null;
        this.sprite = null;

        this._initScene();
    }


    /* Render the background and the logmap

    Parameters:
        renderer (WebGLRenderer): The renderer to use
        cameraOrientation (Quaternion): The orientation of the camera (the logmap sphere will
                                        be rendered using a camera with this orientation, in
                                        order to rotate along the user camera)
    */
    render(renderer, cameraOrientation) {
        // Update the size of the render target if necessary
        if (this.textureSize != this.size * window.devicePixelRatio) {
            this.textureSize = this.size * window.devicePixelRatio;
            this.render_target.setSize(this.textureSize, this.textureSize);
        }

        // Update the logmap using the orientations of the TCP and the target
        this._update(this.target.quaternion, this.robot.getEndEffectorOrientation());

        // Render into the render target
        this.camera.position.x = 0;
        this.camera.position.y = 0;
        this.camera.position.z = 0;
        this.camera.setRotationFromQuaternion(cameraOrientation);
        this.camera.translateZ(10);

        renderer.setRenderTarget(this.render_target);
        renderer.setClearColor(new THREE.Color(0.0, 0.0, 0.0), 0.0);
        renderer.clear();
        renderer.render(this.scene, this.camera);
        renderer.setRenderTarget(null);

        // Render into the DOM element
        renderer.render(this.orthoScene, this.orthoCamera);
    }


    _initScene() {
        const width = this.domElement.clientWidth;
        const height = this.domElement.clientHeight;

        // Cameras
        this.camera = new THREE.PerspectiveCamera(45, 1.0, 0.1, 2000);
        this.camera.up.set(0, 0, 1);

        this.orthoCamera = new THREE.OrthographicCamera(-width / 2, width / 2, height / 2, -height / 2, -10, 10);
        this.orthoCamera.position.z = 10;
        this.orthoCamera.up.set(0, 0, 1);

        // Render target
        this.render_target = new THREE.WebGLRenderTarget(
            this.textureSize, this.textureSize,
            {
                encoding: THREE.sRGBEncoding
            }
        );

        // Scenes
        this.scene = new THREE.Scene();
        this.orthoScene = new THREE.Scene();

        // Sphere
        const sphereGeometry = new THREE.SphereGeometry(1, 32, 16);
        const sphereMaterial = new THREE.MeshPhongMaterial({
            color: 0x1c84b8,
            emissive: 0x072534,
            side: THREE.FrontSide,
            flatShading: false,
            opacity: 0.75,
            transparent: true
        });
        this.sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
        this.scene.add(this.sphere);

        // Points
        const pointGeometry = new THREE.CircleGeometry(0.05, 12);

        const destPointMaterial = new THREE.MeshBasicMaterial({
            color: 0xff0000,
            side: THREE.DoubleSide,
        });

        const srcPointMaterial = new THREE.MeshBasicMaterial({
            color: 0x00ff00,
            side: THREE.DoubleSide,
        });

        const projectedPointMaterial = new THREE.MeshBasicMaterial({
            color: 0xffff00,
            side: THREE.DoubleSide,
        });

        this.destPoint = new THREE.Mesh(pointGeometry, destPointMaterial);
        this.destPoint.position.z = 1.0;

        this.destPointCtrl = new THREE.Object3D();
        this.destPointCtrl.add(this.destPoint);
        this.scene.add(this.destPointCtrl);

        this.srcPoint = new THREE.Mesh(pointGeometry, srcPointMaterial);
        this.srcPoint.position.z = 1.0;

        this.srcPointCtrl = new THREE.Object3D();
        this.srcPointCtrl.add(this.srcPoint);
        this.scene.add(this.srcPointCtrl);

        this.projectedPoint = new THREE.Mesh(pointGeometry, projectedPointMaterial);
        this.scene.add(this.projectedPoint);

        // Projected line
        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0xe97451,
        });

        const points = [new THREE.Vector3(), new THREE.Vector3()];
        this.destPoint.getWorldPosition(points[0]);
        this.projectedPoint.getWorldPosition(points[1]);

        const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);

        this.line = new THREE.Line(lineGeometry, lineMaterial);
        this.scene.add(this.line);

        // Plane
        const planeGeometry = new THREE.PlaneGeometry(3, 3);
        const planeMaterial = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide,
            opacity: 0.3,
            transparent: true
        });

        this.plane = new THREE.Mesh(planeGeometry, planeMaterial);
        this.destPoint.add(this.plane);

        // Lights
        const light = new THREE.HemisphereLight(0xffeeee, 0x111122);
        this.scene.add(light);

        const pointLight = new THREE.PointLight(0xffffff, 60);
        pointLight.position.set(3, -4, 3);
        this.scene.add(pointLight);

        // Sprite in the final scene
        const spriteMaterial = new THREE.SpriteMaterial({
            map: this.render_target.texture,
        });
        this.sprite = new THREE.Sprite(spriteMaterial);
        this.orthoScene.add(this.sprite);

        this._updateSpritePosition();

        // Events handling
        new ResizeObserver(() => this._onDomElementResized()).observe(this.domElement);
    }


    _onDomElementResized() {
        const width = this.domElement.clientWidth;
        const height = this.domElement.clientHeight;

        this.orthoCamera.left = -width / 2;
        this.orthoCamera.right = width / 2;
        this.orthoCamera.top = height / 2;
        this.orthoCamera.bottom = -height / 2;
        this.orthoCamera.updateProjectionMatrix();

        this._updateSpritePosition();
    }


    _updateSpritePosition() {
        const halfWidth = this.domElement.clientWidth / 2;
        const halfHeight = this.domElement.clientHeight / 2;
        const margin = 10;

        if (this.position == 'right') {
            this.sprite.position.set(halfWidth - this.size / 8 - margin, halfHeight - this.size / 8 - margin, 1);
        } else {
            this.sprite.position.set(-halfWidth + this.size / 8 + margin, halfHeight - this.size / 8 - margin - 30, 1);
        }

        this.sprite.scale.set(this.size, this.size, 1);
    }


    _update(mu, f) {
        this.destPointCtrl.setRotationFromQuaternion(mu);
        this.srcPointCtrl.setRotationFromQuaternion(f);

        const base = new THREE.Vector3();
        const y = new THREE.Vector3();

        this.destPoint.getWorldPosition(base);
        this.srcPoint.getWorldPosition(y);

        const temp = y.clone().sub(base.clone().multiplyScalar(base.dot(y)));
        if (temp.lengthSq() > 1e-9) {
            temp.normalize();
            this.projectedPoint.position.addVectors(base, temp.multiplyScalar(this._distance(base, y)));
        } else {
            this.projectedPoint.position.copy(base);
        }

        this.projectedPoint.position.addVectors(base, temp);
        this.destPoint.getWorldQuaternion(this.projectedPoint.quaternion);

        const points = [new THREE.Vector3(), new THREE.Vector3()];
        this.destPoint.getWorldPosition(points[0]);
        this.projectedPoint.getWorldPosition(points[1]);

        this.line.geometry.setFromPoints(points);
    }


    _distance(x, y) {
        let dist = x.dot(y);

        if (dist > 1.0) {
            dist = 1.0;
        } else if (dist < -1.0) {
            dist = -1.0;
        }

        return Math.acos(dist);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



const Shapes = Object.freeze({
    Cube: Symbol("cube"),
    Cone: Symbol("cone"),
    Sphere: Symbol("sphere"),
});



/* Represents a target, an object that can be manipulated by the user that can for example
be used to define a destination position and orientation for the end-effector of the robot.

A target is an Object3D, so you can manipulate it like one.

Note: Targets are top-level objects, so their local position and orientation are also their
position and orientation in world space. 
*/
class Target extends THREE.Object3D {

    /* Constructs the 3D viewer

    Parameters:
        name (str): Name of the target
        position (Vector3): The position of the target
        orientation (Quaternion): The orientation of the target
        color (int/str): Color of the target (by default: 0x0000aa)
        shape (Shapes): Shape of the target (by default: Shapes.Cube)
        listener (function): Function to call when the target is moved/rotated using the mouse
        parameters (dict): Additional shape-dependent parameters (radius, width, height, ...) and opacity
    */
    constructor(name, position, orientation, color=0x0000aa, shape=Shapes.Cube, listener=null, parameters=null) {
        super();

        this.name = name;
        this.listener = listener;

        if (parameters == null)
            parameters = new Map();
        else if (!(parameters instanceof Map))
            parameters = new Map(Object.entries(parameters));

        // Create the mesh
        let geometry = null;
        switch (shape) {
            case Shapes.Cone:
                geometry = new THREE.ConeGeometry(
                    parameters.get('radius') || 0.05,
                    parameters.get('height') || 0.1,
                    12
                );
                break;

            case Shapes.Sphere:
                geometry = new THREE.SphereGeometry(
                    parameters.get('radius') || 0.05
                );
                break;

            case Shapes.Cube:
            default:
                geometry = new THREE.BoxGeometry(
                    parameters.get('width') || 0.1,
                    parameters.get('height') || 0.1,
                    parameters.get('depth') || 0.1
                );
        }

        this.mesh = new THREE.Mesh(
            geometry,
            new THREE.MeshBasicMaterial({
                color: color,
                opacity: parameters.get('opacity') || 0.75,
                transparent: true
            })
        );

        this.mesh.rotateX(Math.PI / 2);
        this.mesh.castShadow = true;
        this.mesh.receiveShadow = false;
        this.mesh.layers = this.layers;

        this.add(this.mesh);

        // Add a wireframe on top of the cone mesh
        const wireframe = new THREE.WireframeGeometry(geometry);

        this.line = new THREE.LineSegments(wireframe);
        this.line.material.depthTest = true;
        this.line.material.opacity = 0.5;
        this.line.material.transparent = true;
        this.line.layers = this.layers;

        this.mesh.add(this.line);

        // Set the target position and orientation
        this.position.copy(position);
        this.quaternion.copy(orientation.clone().normalize());

        this.mesh.tag = 'target-mesh';
        this.tag = 'target';
    }


    /* Returns the position and orientation of the target in an array of the form:
    [px, py, pz, qx, qy, qz, qw]
    */
    transforms() {
        return [
            this.position.x, this.position.y, this.position.z,
            this.quaternion.x, this.quaternion.y, this.quaternion.z, this.quaternion.w,
        ];
    }


    /* Frees the GPU-related resources allocated by this instance. Call this method whenever this
    instance is no longer used in your app.
    */
    dispose() {
        this.mesh.geometry.dispose();
        this.mesh.material.dispose();
        this.line.geometry.dispose();
        this.line.material.dispose();
    }


    _disableVisibility(materials) {
        this.mesh.material.colorWrite = false;
        this.mesh.material.depthWrite = false;

        this.line.material.colorWrite = false;
        this.line.material.depthWrite = false;

        materials.push(this.mesh.material, this.line.material);
    }

}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



class TargetList {

    constructor() {
        this.targets = {};
        this.meshes = [];
    }


    /* Create a new target and add it to the list.

    Parameters:
        name (str): Name of the target
        position (Vector3): The position of the target
        orientation (Quaternion): The orientation of the target
        color (int/str): Color of the target (by default: 0x0000aa)
        shape (Shapes): Shape of the target (by default: Shapes.Cube)
        listener (function): Function to call when the target is moved/rotated using the mouse
        parameters (dict): Additional shape-dependent parameters (radius, width, height, ...)

    Returns:
        The target
    */
    create(name, position, orientation, color, shape=Shapes.Cube, listener=null, parameters=null) {
        const target = new Target(name, position, orientation, color, shape, listener, parameters);
        this.add(target);
        return target;
    }


    /* Add a target to the list.

    Parameters:
        target (Target): The target
    */
    add(target) {
        this.targets[target.name] = target;
        this.meshes.push(target.mesh);
    }


    /* Destroy a target.

    Parameters:
        name (str): Name of the target to destroy
    */
    destroy(name) {
        const target = this.targets[name];
        if (target == undefined)
            return;

        if (target.parent != null)
            target.parent.remove(target);

        const index = this.meshes.indexOf(target.mesh);
        this.meshes.splice(index, 1);

        if (target.dispose != undefined)
            target.dispose();

        delete this.targets[name];
    }


    /* Returns a target.

    Parameters:
        name (str): Name of the target
    */
    get(name) {
        return this.targets[name] || null;
    }

}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



class ObjectList {

    constructor() {
        this.objects = {};
    }


    /* Add an object to the list.

    Parameters:
        object (Object3D): The object
    */
    add(object) {
        this.objects[object.name] = object;
    }


    /* Destroy an object.

    Parameters:
        name (str): Name of the object to destroy
    */
    destroy(name) {
        const object = this.objects[name];
        if (object == undefined)
            return;

        if (object.parent != null)
            object.parent.remove(object);

        if (object.dispose != undefined)
            object.dispose();

        delete this.objects[name];
    }


    /* Returns an object.

    Parameters:
        name (str): Name of the object
    */
    get(name) {
        return this.objects[name] || null;
    }

}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



let cylinderGeometry = null;
let coneGeometry = null;
const axis = new THREE.Vector3();


/* Visual representation of an arrow.

An arrow is an Object3D, so you can manipulate it like one.
*/
class Arrow extends THREE.Object3D {

    /* Constructor

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
    constructor(name, origin, direction, length=1, color=0xffff00, shading=false, headLength=length * 0.2,
        headWidth=headLength * 0.2, radius=headWidth*0.1
    ) {
        super();

        this.name = name;

        if (cylinderGeometry == null) {
            cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 16);
            cylinderGeometry.translate(0, 0.5, 0);

            coneGeometry = new THREE.ConeGeometry(0.5, 1, 16);
            coneGeometry.translate(0, -0.5, 0);
        }

        let material;
        if (shading) {
            material = new THREE.MeshPhongMaterial({
                color: color
            });
        } else {
            material = new THREE.MeshBasicMaterial({
                color: color
            });
        }

        this.cylinder = new THREE.Mesh(cylinderGeometry, material);
        this.cylinder.layers = this.layers;

        this.add(this.cylinder);

        this.cone = new THREE.Mesh(coneGeometry, material);
        this.cone.layers = this.layers;

        this.add(this.cone);

        this.position.copy(origin);
        this.setDirection(direction);
        this.setDimensions(length, headLength, headWidth, radius);
    }


    /* Sets the direction of the arrow
    */
    setDirection(direction) {
        // 'direction' is assumed to be normalized
        if (direction.y > 0.99999) {
            this.quaternion.set(0, 0, 0, 1);
        } else if (direction.y < - 0.99999) {
            this.quaternion.set(1, 0, 0, 0);
        } else {
            axis.set(direction.z, 0, -direction.x).normalize();
            const radians = Math.acos(direction.y);
            this.quaternion.setFromAxisAngle(axis, radians);
        }
    }


    /* Sets the dimensions of the arrow

    Parameters:
        length (Number): Length of the arrow
        headLength (Number): The length of the head of the arrow (default is 0.2 * length)
        headWidth (Number): The width of the head of the arrow (default is 0.2 * headLength)
        radius (Number): The radius of the line part of the arrow (default is 0.1 * headWidth)
    */
    setDimensions(length, headLength=length * 0.2, headWidth=headLength * 0.2, radius=headWidth*0.3) {

        this.cylinder.scale.set(
            Math.max(0.0001, radius),
            Math.max(0.0001, length - headLength),
            Math.max(0.0001, radius)
        );

        this.cylinder.updateMatrix();

        this.cone.scale.set(headWidth, headLength, headWidth);
        this.cone.position.y = length;
        this.cone.updateMatrix();
    }


    /* Sets the color of the arrow
    */
    setColor(color) {
        this.line.material.color.set(color);
        this.cone.material.color.set(color);
    }


    /* Frees the GPU-related resources allocated by this instance. Call this method whenever this
    instance is no longer used in your app.
    */
    dispose() {
        this.cylinder.geometry.dispose();
        this.cone.geometry.dispose();
        this.cone.material.dispose();
    }


    _disableVisibility(materials) {
        this.cylinder.material.colorWrite = false;
        this.cylinder.material.depthWrite = false;

        this.cone.material.colorWrite = false;
        this.cone.material.depthWrite = false;

        materials.push(this.cylinder.material, this.cone.material);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Visual representation of a path.

A path is an Object3D, so you can manipulate it like one.
*/
class Path extends THREE.Object3D {

    /* Constructor

    Parameters:
        name (str): Name of the path
        points (list or Vector3/list of lists of 3 numbers): Points defining the path
        radius (Number): The radius of the path (default is 0.01)
        color (int/str): Color of the path (by default: 0xffff00)
        shading (bool): Indicates if the path must be affected by lights (by default: false)
        transparent (bool): Indicates if the path must be transparent (by default: false)
        opacity (Number): Opacity level for transparent paths (between 0 and 1, default: 0.5)
    */
    constructor(name, points, radius=0.01, color=0xffff00, shading=false, transparent=false, opacity=0.5) {
        super();

        this.name = name;

        let curvePoints = points;
        if (!(points[0] instanceof THREE.Vector3)) {
            curvePoints = points.map(x => new THREE.Vector3(x[0], x[1], x[2]));
        }

        const curve = new THREE.CatmullRomCurve3(curvePoints);

        const geometry = new THREE.TubeGeometry(curve, points.length * 2, radius, 16, false);

        let material;
        if (shading) {
            material = new THREE.MeshPhongMaterial({
                color: color,
                opacity: opacity,
                transparent: transparent
            });
        } else {
            material = new THREE.MeshBasicMaterial({
                color: color,
                opacity: opacity,
                transparent: transparent
            });
        }

        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.layers = this.layers;

        this.add(this.mesh);

        const sphereGeometry = new THREE.SphereGeometry(radius, 16, 16);//, 0, Math.PI);

        this.startSphere = new THREE.Mesh(sphereGeometry, material);
        this.startSphere.position.copy(curvePoints[0]);
        this.startSphere.layers = this.layers;

        this.add(this.startSphere);

        this.endSphere = new THREE.Mesh(sphereGeometry, material);
        this.endSphere.position.copy(curvePoints[curvePoints.length - 1]);
        this.endSphere.layers = this.layers;

        this.add(this.endSphere);
    }


    /* Frees the GPU-related resources allocated by this instance. Call this method whenever this
    instance is no longer used in your app.
    */
    dispose() {
        this.mesh.geometry.dispose();
        this.startSphere.geometry.dispose();
        this.endSphere.geometry.dispose();
        this.mesh.material.dispose();
    }


    _disableVisibility(materials) {
        this.mesh.material.colorWrite = false;
        this.mesh.material.depthWrite = false;

        this.startSphere.material.colorWrite = false;
        this.startSphere.material.depthWrite = false;

        this.endSphere.material.colorWrite = false;
        this.endSphere.material.depthWrite = false;

        materials.push(this.mesh.material, this.startSphere.material, this.endSphere.material);
    }

}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Visual representation of a path.

A path is an Object3D, so you can manipulate it like one.
*/
class Point extends THREE.Object3D {

    /* Constructor

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
    constructor(name, position, radius=0.01, color=0xffff00, label=null, shading=false,
                transparent=false, opacity=0.5) {
        super();

        this.name = name;
        this.position.copy(position);

        if (typeof(color) == 'string')
            if (color[0] == '#')
                color = color.substring(1);
            color = Number('0x' + color);

        color = new THREE.Color(color);

        const geometry = new THREE.SphereGeometry(radius);

        let material;
        if (shading) {
            material = new THREE.MeshPhongMaterial({
                color: color,
                opacity: opacity,
                transparent: transparent
            });
        } else {
            material = new THREE.MeshBasicMaterial({
                color: color,
                opacity: opacity,
                transparent: transparent
            });
        }

        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.layers = this.layers;

        this.add(this.mesh);

        if (label != null) {
            this.labelElement = document.createElement('div');
            this.labelElement.style.fontSize = '1vw';

            katex.render(String.raw`\color{#` + color.getHexString() + `}` + label, this.labelElement, {
                throwOnError: false
            });

            this.label = new CSS2DObject(this.labelElement);
            this.label.position.set(0,2 * radius + 0.01, 0);

            this.label.layers.disableAll();
            this.label.layers.enable(31);

            this.add(this.label);
        }
    }


    setTexture(url) {
        const texture = new THREE.TextureLoader().load(url);
        this.mesh.material.map = texture;
    }


    /* Frees the GPU-related resources allocated by this instance. Call this method whenever this
    instance is no longer used in your app.
    */
    dispose() {
        this.mesh.geometry.dispose();
        this.mesh.material.dispose();
    }


    _disableVisibility(materials) {
        this.mesh.material.colorWrite = false;
        this.mesh.material.depthWrite = false;

        materials.push(this.mesh.material);
    }

}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



let sphereGeometry = null;


/* Computes the covariance matrix of a gaussian from an orientation and scale

Parameters:
    quaternion (Quaternion): The orientation
    scale (Vector3): The scale
*/
function sigmaFromQuaternionAndScale(quaternion, scale) {
    let rot4x4 = new THREE.Matrix4().makeRotationFromQuaternion(quaternion);

    let RG = new THREE.Matrix3().set(
            scale.x * rot4x4.elements[0], scale.y * rot4x4.elements[4], scale.z * rot4x4.elements[8],
            scale.x * rot4x4.elements[1], scale.y * rot4x4.elements[5], scale.z * rot4x4.elements[9],
            scale.x * rot4x4.elements[2], scale.y * rot4x4.elements[6], scale.z * rot4x4.elements[10]
    );

    let sigma = new THREE.Matrix3().copy(RG);

    RG.transpose();
    sigma.multiply(RG);

    return sigma;
}


/* Computes the covariance matrix of a gaussian from a rotation and scaling matrix
*/
function sigmaFromMatrix3(matrix) {
    let RG = new THREE.Matrix3().copy(matrix);

    let sigma = new THREE.Matrix3().copy(RG);

    RG.transpose();
    sigma.multiply(RG);

    return sigma;
}


/* Computes the covariance matrix of a gaussian from the rotation and scaling parts of a matrix
(the upper 3x3 part)
*/
function sigmaFromMatrix4(matrix) {
    let RG = new THREE.Matrix3().setFromMatrix4(matrix);

    let sigma = new THREE.Matrix3().copy(RG);

    RG.transpose();
    sigma.multiply(RG);

    return sigma;
}


/* Computes the rotation and scaling matrix corresponding to the covariance matrix of a gaussian
*/
function matrixFromSigma(sigma) {
    const sigma2 = math.reshape(math.matrix(sigma.elements), [3, 3]);

    const ans = math.eigs(sigma2);

    // Here we do RG = V * diagmat(sqrt(d))
    // where 'd' is a vector of eigenvalues and 'V' a matrix where each column contains an
    // eigenvector
    const d = math.diag(math.map(ans.values, math.sqrt));

    const V = math.matrixFromColumns(
        ans.eigenvectors[0].vector,
        ans.eigenvectors[1].vector,
        ans.eigenvectors[2].vector
    );

    const RG = math.multiply(V, d);

    return new THREE.Matrix3().fromArray(math.flatten(math.transpose(RG)).toArray());
}



/* Visual representation of a gaussian.

A gaussian is an Object3D, so you can manipulate it like one. Modifying the orientation and scale
of the gaussian modifies its covariance matrix.
*/
class Gaussian extends THREE.Object3D {

    /* Constructor

    Parameters:
        name (str): Name of the arrow
        mu (Vector3): Position of the gaussian
        sigma (Matrix): Covariance matrix of the gaussian
        color (int/str): Color of the gaussian (by default: 0xffff00)
        listener (function): Function to call when the gaussian is modified using the mouse
    */
    constructor(name, mu, sigma, color=0xffff00, listener=null) {
        super();

        this.name = name;
        this.listener = listener;

        if (sphereGeometry == null) {
            sphereGeometry = new THREE.SphereGeometry(1.0, 32, 16);
        }

        let material = new THREE.ShaderMaterial({
            uniforms: {
                color: { value: new THREE.Color(color) },
                mu: { value: mu },
                invSigma: { value: new THREE.Matrix3().copy(sigma).invert() },
            },
            transparent: true,
            depthWrite: true,
            vertexShader: `
                varying vec3 positionCameraSpace;

                void main() {
                    positionCameraSpace = (modelViewMatrix * vec4(position, 1.0)).xyz;
                    gl_Position = projectionMatrix * vec4(positionCameraSpace, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 color;
                uniform vec3 mu;
                uniform mat3 invSigma;
                uniform mat4 modelViewMatrix;

                varying vec3 positionCameraSpace;

                void main() {
                    vec3 eye_dir = normalize(positionCameraSpace);

                    vec3 dir_step = eye_dir * 0.001;

                    vec3 position = positionCameraSpace;

                    vec3 muCameraSpace = (modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;

                    float maxAlpha = 0.0f;

                    for (int i = 0; i < 500; ++i) {
                        vec3 e = position - muCameraSpace;

                        float alpha = clamp(exp(-(invSigma[0][0] * e.x * e.x + 2.0 * invSigma[0][1] * e.x * e.y +
                                                  invSigma[1][1] * e.y * e.y + 2.0 * invSigma[0][2] * e.x * e.z +
                                                  invSigma[2][2] * e.z * e.z + 2.0 * invSigma[1][2] * e.y * e.z) * 2.0),
                                            0.0, 1.0
                        );

                        if (alpha > maxAlpha)
                            maxAlpha = alpha;

                        // Stop when the alpha becomes significantly smaller than the maximum
                        // value seen so far
                        else if (alpha < maxAlpha * 0.9f)
                            break;

                        // Stop when the alpha becomes very large
                        if (maxAlpha >= 0.999f)
                            break;

                        position = position + dir_step;
                    }

                    gl_FragColor = vec4(color, pow(maxAlpha, 1.0 / 2.2));
                }
            `,
        });

        this.sphere = new THREE.Mesh(sphereGeometry, material);
        this.sphere.layers = this.layers;

        this.add(this.sphere);

        this.position.copy(mu);
        this.setSigma(sigma);

        this.center = new THREE.Mesh(
            new THREE.SphereGeometry(0.1),
            new THREE.MeshBasicMaterial({
                visible: true,
                color: 0x000000
            })
        );

        this.center.tag = 'gaussian-center';
        this.center.gaussian = this;

        // this.add(this.center);
    }


    /* Returns the covariance matrix of the gaussian
    */
    sigma() {
        this.updateMatrixWorld();
        return sigmaFromMatrix4(this.matrixWorld);
    }


    /* Sets the covariance matrix of the gaussian
    */
    setSigma(sigma) {
        const m = matrixFromSigma(sigma);

        const transforms = new THREE.Matrix4().setFromMatrix3(m);
        transforms.setPosition(this.position);

        this.position.set(0.0, 0.0, 0.0);
        this.quaternion.set(0.0, 0.0, 0.0, 1.0);
        this.scale.set(1.0, 1.0, 1.0);

        this.applyMatrix4(transforms);
    }


    /* Sets the color of the gaussian
    */
    setColor(color) {
        this.sphere.material.color.set(color);
    }


    /* Frees the GPU-related resources allocated by this instance. Call this method whenever this
    instance is no longer used in your app.
    */
    dispose() {
        this.sphere.geometry.dispose();
        this.sphere.material.dispose();
    }


	raycast(raycaster, intersects) {
        this.center.position.copy(this.position);
        this.center.updateMatrixWorld();

        return this.center.raycast(raycaster, intersects);
    }


    _update(viewMatrix) {
        if (this.sphere.material.uniforms == undefined)
            return;

        let sigma = this.sigma();
        sigma.premultiply(viewMatrix);
        sigma.multiply(new THREE.Matrix3().copy(viewMatrix).transpose());

        this.sphere.material.uniforms['invSigma'].value = sigma.invert();
    }


    _disableVisibility(materials) {
        this.sphere.material.colorWrite = false;
        this.sphere.material.depthWrite = false;

        materials.push(this.sphere.material);
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Truncated cone for haze rendering
*/
class HazeGeometry extends THREE.BufferGeometry {

    constructor(nbSlices, proportion, color) {
        super();

        this.type = 'HazeGeometry';

        // buffers
        const indices = [];
        const vertices = [];
        const colors = [];

        // Compute elevation h for transparancy transition point
        const alpha = Math.atan2(1.0, proportion);
        const beta = (0.75 * Math.PI) - alpha;
        const h = Math.sqrt(0.5) * proportion * Math.sin(alpha) / Math.sin(beta);

        for (let i = 0; i < 2; ++i) {
            const h1 = (i == 0 ? 0 : h);
            const h2 = (i == 0 ? h : 1);

            const alpha1 = (i == 1);
            const alpha2 = (i == 0);

            for (let j = 0; j < nbSlices; ++j) {
                const az1 = (2.0 * Math.PI * (j + 0)) / nbSlices;
                const az2 = (2.0 * Math.PI * (j + 1)) / nbSlices;

                const index = vertices.length / 3;

                this._makeVertex(vertices, az1, h1, proportion);
                this._makeVertex(vertices, az2, h1, proportion);
                this._makeVertex(vertices, az2, h2, proportion);
                this._makeVertex(vertices, az1, h2, proportion);

                colors.push(color.r, color.g, color.b, alpha1);
                colors.push(color.r, color.g, color.b, alpha1);
                colors.push(color.r, color.g, color.b, alpha2);
                colors.push(color.r, color.g, color.b, alpha2);

                indices.push(index, index + 1, index + 2);
                indices.push(index + 2, index + 3, index);
            }
        }

        // build geometry
        this.setIndex(indices);
        this.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        this.setAttribute('color', new THREE.Float32BufferAttribute(colors, 4));
    }

    _makeVertex(vertices, az, h, r) {
        vertices.push(
            Math.cos(az) * (1.0 - r * (1.0 - h)),
            h,
            -Math.sin(az) * (1.0 - r * (1.0 - h))
        );
    }

}



/* Truncated cone for haze rendering
*/
class Haze extends THREE.Object3D {

    constructor(nbSlices, proportion, color) {
        super();

        const geometry = new HazeGeometry(nbSlices, proportion, color);

        const material = new THREE.MeshBasicMaterial({
            transparent: true,
            vertexColors: true,
            side: THREE.BackSide
        });

        this.mesh = new THREE.Mesh(geometry, material);
        this.add(this.mesh);
    }

}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */


let toonGradientMap = null;


function buildGradientMap(nbColors, maxValue=255) {
    const colors = new Uint8Array(nbColors);

    for (let c = 0; c <= nbColors; c++)
        colors[c] = (c / nbColors) * maxValue;

    const gradientMap = new THREE.DataTexture(colors, nbColors, 1, THREE.RedFormat);
    gradientMap.needsUpdate = true;

    return gradientMap;
}


function enableToonShading(object, gradientMap=null) {
    if (gradientMap == null) {
        if (toonGradientMap == null)
            toonGradientMap = buildGradientMap(3);

        gradientMap = toonGradientMap;
    }

    if (object.isMesh) {
        object.material = new THREE.MeshToonMaterial({
            color: object.material.color,
            gradientMap: gradientMap,
        });
    } else {
        object.children.forEach(child => { enableToonShading(child, gradientMap); });
    }
}


function enableLightToonShading(object, gradientMap=null) {
    if (gradientMap == null)
        gradientMap = buildGradientMap(3, 128);

    if (object.isMesh) {
        object.material.color.r = object.material.color.r * 0.7;
        object.material.color.g = object.material.color.g * 0.7;
        object.material.color.b = object.material.color.b * 0.7;

        object.material = new THREE.MeshToonMaterial({
            color: object.material.color,
            emissive: 0xaaaaaa,
            gradientMap: gradientMap,
        });
    } else {
        object.children.forEach(child => { enableLightToonShading(child, gradientMap); });
    }
}


function modifyMaterialColor(object, color) {
    if (object.isMesh) {
        const hsl = {};
        object.material.color.getHSL(hsl);

        object.material.color.r = color.r * hsl.l;
        object.material.color.g = color.g * hsl.l;
        object.material.color.b = color.b * hsl.l;
    } else {
        object.children.forEach(child => { modifyMaterialColor(child, color); });
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */


/* Allows to retrieve informations about a specific body in the physics simulation.
*/
class PhysicalBody {

    constructor(name, bodyId, physicsSimulator) {
        this.name = name;
        this.bodyId = bodyId;
        this.physicsSimulator = physicsSimulator;
    }


    position() {
        return this.physicsSimulator.getBodyPosition(this.bodyId);
    }


    setPosition(position) {
        return this.physicsSimulator.setBodyPosition(this.bodyId, position);
    }


    orientation() {
        return this.physicsSimulator.getBodyOrientation(this.bodyId);
    }


    setOrientation(orientation) {
        return this.physicsSimulator.setBodyOrientation(this.bodyId, orientation);
    }

}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */


/* Manages the toolbar at the top of the canvas
*/
class Toolbar {

    /* Construct the toolbar.

    Parameters:
        domElement (element): The DOM element used by the 3D viewer
    */
    constructor(domElement) {
        this.container = document.createElement('div');
        this.container.className = 'toolbar';
        domElement.appendChild(this.container);
    }


    /* Add a section in the toolbar and returns it (a div)
    */
    addSection() {
        const section = document.createElement('div');
        section.className = 'section';
        this.container.appendChild(section);

        return section;
    }
}


class GripperToolbarSection {

    /* Construct the toolbar.

    Parameters:
        toolbar (Toolbar): The toolbar to which the section must be added
        robot (Robot): The robot with the gripper we must manipulate
    */
    constructor(toolbar, robot) {
        this.section = toolbar.addSection();
        this.robot = robot;

        this.btn = document.createElement('button');
        this.btn.className = 'round';
        this.btn.innerText = 'Open gripper';
        this.section.appendChild(this.btn);

        this.enabled = true;

        this.btn.onclick = (evt) => { robot.toggleGripper(); };
    }


    destroy() {
        this.section.remove();
    }


    update(isClosed) {
        this.enabled = true;
        this.btn.disabled = false;
        this.btn.classList.remove('disabled');

        if (isClosed)
            this.btn.innerText = 'Open gripper';
        else
            this.btn.innerText = 'Close gripper';
    }


    disable() {
        this.enabled = false;
        this.btn.disabled = true;
        this.btn.classList.add('disabled');
    }


    isEnabled() {
        return this.enabled;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 * SPDX-FileCopyrightText: Copyright © 2022 Nikolas Dahn
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 * SPDX-FileContributor: Nikolas Dahn
 *
 * SPDX-License-Identifier: MIT
 *
 * This file is a modification of the one implemented in https://github.com/ndahn/Rocksi
 *
 */

class RobotConfiguration {
    constructor() {
        // Root link of the robot
        this.robotRoot = null;

        this.tools = [
            // Each entry should have this structure
            // {
            //     root: null,              // Root link of the tool of the robot (can be null if no tool)
            //     tcpSite: null,           // Site to use as the TCP
            //     tcpSize: 0.1,            // Size of the TCP collision object
            //     type: null,              // Type of the tool ("generic", "gripper")
            //
            // For grippers:
            //     buttonOffset: [0, 0, 0], // Location of the button, relative to the tcp
            //     ignoredActuators: [],    // Names of the actuators to ignore when opening/closing
            //     invertedActuators: [],   // Names of the actuators to invert when opening/closing
            // }
        ];

        // Default pose of the robot
        this.defaultPose = {
        },

        this.jointPositionHelpers = {
            // Offsets to apply to the joint position helpers
            offsets: {},

            // Joint position helpers that must be inverted
            inverted: [],
        };
    }


    addPrefix(prefix) {
        const configuration = new RobotConfiguration();

        configuration.robotRoot = prefix + this.robotRoot;

        for (const tool of this.tools) {
            let tool2 = {
                root: (tool.root != null ? prefix + tool.root : null),
                tcpSite: (tool.tcpSite != null ? prefix + tool.tcpSite : null),
                tcpSize: tool.tcpSize,
                type: tool.type,
                buttonOffset: tool.buttonOffset,
            };

            configuration.tools.push(tool2);
        }

        for (let name in this.defaultPose)
            configuration.defaultPose[prefix + name] = this.defaultPose[name];

        for (let name in this.jointPositionHelpers.offsets)
            configuration.jointPositionHelpers.offsets[prefix + name] = this.jointPositionHelpers.offsets[name];

        for (let name of this.jointPositionHelpers.inverted)
            configuration.jointPositionHelpers.inverted.push(prefix + name);

        return configuration;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 * SPDX-FileCopyrightText: Copyright © 2022 Nikolas Dahn
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 * SPDX-FileContributor: Nikolas Dahn
 *
 * SPDX-License-Identifier: MIT
 *
 * This file is a modification of the one implemented in https://github.com/ndahn/Rocksi
 *
 */



class PandaConfiguration extends RobotConfiguration {
    constructor() {
        super();

        this.robotRoot = "link0";

        this.tools = [
            {
                root: "hand",
                tcpSite: "tcp",
                tcpSize: 0.1,
                type: "gripper",
                buttonOffset: [0, -0.11, 0.05,]
            }
        ];

        this.defaultPose = {
            joint1: 0.5,
            joint2: -0.3,
            joint4: -1.8,
            joint6: 1.5,
            joint7: 1.0,
        };

        this.jointPositionHelpers.offsets = {
            joint1: -0.19,
            joint3: -0.12,
            joint5: -0.26,
            joint6: -0.015,
            joint7: 0.05,
        };

        this.jointPositionHelpers.inverted = [
            'joint4',
            'joint5',
            'joint6',
        ];
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 * SPDX-FileCopyrightText: Copyright © 2022 Nikolas Dahn
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 * SPDX-FileContributor: Nikolas Dahn
 *
 * SPDX-License-Identifier: MIT
 *
 * This file is a modification of the one implemented in https://github.com/ndahn/Rocksi
 *
 */



class PandaNoHandConfiguration extends RobotConfiguration {
    constructor() {
        super();

        this.robotRoot = "link0";

        this.tools = [
            {
                root: null,
                tcpSite: "attachment_site",
                type: "generic",
            }
        ];

        this.defaultPose = {
            joint1: 0.5,
            joint2: -0.3,
            joint4: -1.8,
            joint6: 1.5,
            joint7: 1.0,
        };

        this.jointPositionHelpers.offsets = {
            joint1: -0.19,
            joint3: -0.12,
            joint5: -0.26,
            joint6: -0.015,
            joint7: 0.05,
        };

        this.jointPositionHelpers.inverted = [
            'joint4',
            'joint5',
            'joint6',
        ];
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 */



class G1Configuration extends RobotConfiguration {
    constructor() {
        super();

        this.robotRoot = "pelvis";

        this.tools = [
            {
                root: "right_hand",
                tcpSite: "right_tcp",
                tcpSize: 0.08,
                type: "generic",
            },
            {
                root: "left_hand",
                tcpSite: "left_tcp",
                tcpSize: 0.08,
                type: "generic",
            },
            {
                root: "right_foot",
                tcpSite: "right_foot_tcp",
                tcpSize: 0.08,
                type: "generic",
            },
            {
                root: "left_foot",
                tcpSite: "left_foot_tcp",
                tcpSize: 0.08,
                type: "generic",
            },
        ];
    }
}


class G1UpperBodyConfiguration extends RobotConfiguration {
    constructor() {
        super();

        this.robotRoot = "pelvis";

        this.tools = [
            {
                root: "right_hand",
                tcpSite: "right_tcp",
                tcpSize: 0.08,
                type: "generic",
            },
            {
                root: "left_hand",
                tcpSite: "left_tcp",
                tcpSize: 0.08,
                type: "generic",
            },
        ];
    }
}


class G1WithHandsConfiguration extends RobotConfiguration {
    constructor() {
        super();

        this.robotRoot = "pelvis";

        this.tools = [
            {
                root: "right_hand",
                tcpSite: "right_hand_tcp",
                tcpSize: 0.08,
                type: "gripper",
                ignoredActuators: ['right_hand_thumb_0_joint'],
                invertedActuators: [
                    'right_hand_index_0_joint', 'right_hand_index_1_joint',
                    'right_hand_middle_0_joint', 'right_hand_middle_1_joint'
                ],
                buttonOffset: [0.1, -0.07, 0],
                state: "opened",
            },
            {
                root: "left_hand",
                tcpSite: "left_hand_tcp",
                tcpSize: 0.08,
                type: "gripper",
                ignoredActuators: ['left_hand_thumb_0_joint'],
                invertedActuators: ['left_hand_thumb_1_joint', 'left_hand_thumb_2_joint'],
                buttonOffset: [0.1, 0.07, 0],
                state: "opened",
            },
            {
                root: "right_foot",
                tcpSite: "right_foot_tcp",
                tcpSize: 0.08,
                type: "generic",
            },
            {
                root: "left_foot",
                tcpSite: "left_foot_tcp",
                tcpSize: 0.08,
                type: "generic",
            },
        ];
    }
}


class G1WithHandsUpperBodyConfiguration extends RobotConfiguration {
    constructor() {
        super();

        this.robotRoot = "pelvis";

        this.tools = [
            {
                root: "right_hand",
                tcpSite: "right_hand_tcp",
                tcpSize: 0.08,
                type: "gripper",
                ignoredActuators: ['right_hand_thumb_0_joint'],
                invertedActuators: [
                    'right_hand_index_0_joint', 'right_hand_index_1_joint',
                    'right_hand_middle_0_joint', 'right_hand_middle_1_joint'
                ],
                buttonOffset: [0.1, -0.07, 0],
                state: "opened",
            },
            {
                root: "left_hand",
                tcpSite: "left_hand_tcp",
                tcpSize: 0.08,
                type: "gripper",
                ignoredActuators: ['left_hand_thumb_0_joint'],
                invertedActuators: ['left_hand_thumb_1_joint', 'left_hand_thumb_2_joint'],
                buttonOffset: [0.1, 0.07, 0],
                state: "opened",
            },
        ];
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Download all the files needed to simulate and display the Franka Emika Panda robot

The files are stored in Mujoco's filesystem at '/scenes/franka_emika_panda'
*/
async function downloadPandaRobot() {
    const dstFolder = '/scenes/franka_emika_panda';
    const srcURL = getURL('models/franka_emika_panda/');

    await downloadFiles(
        dstFolder,
        srcURL,
        [
            'panda.xml',
            'panda_nohand.xml'
        ]
    );

    await downloadFiles(
        dstFolder + '/assets',
        srcURL + 'assets/',
        [
            'link0.stl',
            'link0.stl',
            'link1.stl',
            'link2.stl',
            'link3.stl',
            'link4.stl',
            'link5_collision_0.obj',
            'link5_collision_1.obj',
            'link5_collision_2.obj',
            'link6.stl',
            'link7.stl',
            'hand.stl',
            'link0_0.obj',
            'link0_1.obj',
            'link0_2.obj',
            'link0_3.obj',
            'link0_4.obj',
            'link0_5.obj',
            'link0_7.obj',
            'link0_8.obj',
            'link0_9.obj',
            'link0_10.obj',
            'link0_11.obj',
            'link1.obj',
            'link2.obj',
            'link3_0.obj',
            'link3_1.obj',
            'link3_2.obj',
            'link3_3.obj',
            'link4_0.obj',
            'link4_1.obj',
            'link4_2.obj',
            'link4_3.obj',
            'link5_0.obj',
            'link5_1.obj',
            'link5_2.obj',
            'link6_0.obj',
            'link6_1.obj',
            'link6_2.obj',
            'link6_3.obj',
            'link6_4.obj',
            'link6_5.obj',
            'link6_6.obj',
            'link6_7.obj',
            'link6_8.obj',
            'link6_9.obj',
            'link6_10.obj',
            'link6_11.obj',
            'link6_12.obj',
            'link6_13.obj',
            'link6_14.obj',
            'link6_15.obj',
            'link6_16.obj',
            'link7_0.obj',
            'link7_1.obj',
            'link7_2.obj',
            'link7_3.obj',
            'link7_4.obj',
            'link7_5.obj',
            'link7_6.obj',
            'link7_7.obj',
            'hand_0.obj',
            'hand_1.obj',
            'hand_2.obj',
            'hand_3.obj',
            'hand_4.obj',
            'finger_0.obj',
            'finger_1.obj',
        ]
    );
}



/* Download all the files needed to simulate and display the Unitree G1 robot

The files are stored in Mujoco's filesystem at '/scenes/unitree_g1'
*/
async function downloadG1Robot() {
    const dstFolder = '/scenes/unitree_g1';
    const srcURL = getURL('models/unitree_g1/');

    await downloadFiles(
        dstFolder,
        srcURL,
        [
            'g1.xml',
            'g1_upperbody.xml',
            'g1_with_hands.xml',
            'g1_with_hands_upperbody.xml',
        ]
    );

    await downloadFiles(
        dstFolder + '/assets',
        srcURL + 'assets/',
        [
            'head_link.STL',
            'left_ankle_pitch_link.STL',
            'left_ankle_roll_link.STL',
            'left_elbow_link.STL',
            'left_hand_index_0_link.STL',
            'left_hand_index_1_link.STL',
            'left_hand_middle_0_link.STL',
            'left_hand_middle_1_link.STL',
            'left_hand_palm_link.STL',
            'left_hand_thumb_0_link.STL',
            'left_hand_thumb_1_link.STL',
            'left_hand_thumb_2_link.STL',
            'left_hip_pitch_link.STL',
            'left_hip_roll_link.STL',
            'left_hip_yaw_link.STL',
            'left_knee_link.STL',
            'left_rubber_hand.STL',
            'left_shoulder_pitch_link.STL',
            'left_shoulder_roll_link.STL',
            'left_shoulder_yaw_link.STL',
            'left_wrist_pitch_link.STL',
            'left_wrist_roll_link.STL',
            'left_wrist_yaw_link.STL',
            'logo_link.STL',
            'pelvis.STL',
            'pelvis_contour_link.STL',
            'right_ankle_pitch_link.STL',
            'right_ankle_roll_link.STL',
            'right_elbow_link.STL',
            'right_hand_index_0_link.STL',
            'right_hand_index_1_link.STL',
            'right_hand_middle_0_link.STL',
            'right_hand_middle_1_link.STL',
            'right_hand_palm_link.STL',
            'right_hand_thumb_0_link.STL',
            'right_hand_thumb_1_link.STL',
            'right_hand_thumb_2_link.STL',
            'right_hip_pitch_link.STL',
            'right_hip_roll_link.STL',
            'right_hip_yaw_link.STL',
            'right_knee_link.STL',
            'right_rubber_hand.STL',
            'right_shoulder_pitch_link.STL',
            'right_shoulder_roll_link.STL',
            'right_shoulder_yaw_link.STL',
            'right_wrist_pitch_link.STL',
            'right_wrist_roll_link.STL',
            'right_wrist_yaw_link.STL',
            'torso_link_rev_1_0.STL',
            'waist_roll_link_rev_1_0.STL',
            'waist_yaw_link_rev_1_0.STL',
        ]
    );
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 */



class RobotBuilder {

    constructor(name) {
        this.name = name;               // Name of the robot (must be unique in the scene)
        this.q = [];                    // Angle about previous z
        this.q_offset = [];             // Offset on articulatory joints
        this.alpha = [];                // Angle about common normal
        this.d = [];                    // Offset along previous z to the common normal
        this.r = [];                    // Length of the common normal
        this.defaultPose= [];           // Default pose of the robots
        this.colors = [];               // As many as you want, links will alternate between those colors
        this.position = [0, 0, 0];      // Position of the robot
        this.quaternion = [1, 0, 0, 0]; // Orientation of the robot, as a quaternion (w, x, y, z)
    }

    clone(name) {
        const builder = new RobotBuilder(name);

        builder.q = [ ...this.q];
        builder.q_offset = [ ...this.q_offset];
        builder.alpha = [ ...this.alpha];
        builder.d = [ ...this.d];
        builder.r = [ ...this.r];
        builder.defaultPose = [ ...this.defaultPose];
        builder.colors = [ ...this.colors];
        builder.position = [ ...this.position];
        builder.quaternion = [ ...this.quaternion];

        return builder;
    }

    generateXMLDocument() {
        const xmlDoc = new Document();

        const xmlRoot = xmlDoc.createElement("mujoco");
        xmlRoot.setAttribute("model", this.name);
        xmlDoc.append(xmlRoot);

        const xmlCompiler = xmlDoc.createElement("compiler");
        xmlCompiler.setAttribute("autolimits", "true");
        xmlRoot.append(xmlCompiler);

        const xmlOption = xmlDoc.createElement("option");
        xmlOption.setAttribute("integrator", "implicit");
        xmlRoot.append(xmlOption);

        const xmlDefaults = this._createDefaults(xmlDoc);
        xmlRoot.append(xmlDefaults);

        const xmlWorldBody = xmlDoc.createElement("worldbody");
        xmlRoot.append(xmlWorldBody);

        const xmlActuators = xmlDoc.createElement("actuator");
        xmlRoot.append(xmlActuators);

        let parent = xmlWorldBody;

        const [f, R] = this._fkin([0, 0, 0, 0, 0, 0, 0]);

        for (let n = 0; n < this.q.length + 1; ++n) {
            const xmlBody = this._createXmlBody(
                xmlDoc,
                this.name + "_link" + n,
                f, R, n,
                (n > 0) && (n < this.q.length) ? this.name + "_joint" + n : null,
            );
            parent.append(xmlBody);

            if ((n > 0) && (n < this.q.length)) {
                const xmlGeneral = xmlDoc.createElement("general");
                xmlGeneral.setAttribute("name", this.name + "_actuator" + n);
                xmlGeneral.setAttribute("joint", this.name + "_joint" + n);
                xmlGeneral.setAttribute("gainprm", "4500");
                xmlGeneral.setAttribute("biasprm", "0 -4500 -450");
                xmlActuators.append(xmlGeneral);
            }

            parent = xmlBody;
        }

        const xmlSite = xmlDoc.createElement("site");
        xmlSite.setAttribute("name", this.name + "_tcp");
        xmlSite.setAttribute("pos", "0 0 0");
        xmlSite.setAttribute("size", "0.001");
        parent.append(xmlSite);

        return xmlDoc;
    }

    getConfiguration() {
        const config = new RobotConfiguration();

        config.robotRoot = this.name + "_link0";
        config.tcpSite = this.name + "_tcp";
        config.tcpSize = 0.04;

        config.defaultPose = {};

        for (let i = 0; i < this.defaultPose.length; ++i)
            config.defaultPose[this.name + "_joint" + (i + 1)] = this.defaultPose[i];

        if (this.prefix != null)
            config.addPrefix(this.prefix);

        return config;
    }

    _createDefaults(xmlDoc) {
        const xmlDefaults = xmlDoc.createElement("default");

        const xmlVisualDefault = xmlDoc.createElement("default");
        xmlVisualDefault.setAttribute("class", this.name + "_visual");
        xmlDefaults.append(xmlVisualDefault);

        const xmlVisualGeomDefault = xmlDoc.createElement("geom");
        xmlVisualGeomDefault.setAttribute("contype", "0");
        xmlVisualGeomDefault.setAttribute("conaffinity", "0");
        xmlVisualGeomDefault.setAttribute("group", "2");
        xmlVisualDefault.append(xmlVisualGeomDefault);

        const xmlCollisionDefault = xmlDoc.createElement("default");
        xmlCollisionDefault.setAttribute("class", this.name + "_collision");
        xmlDefaults.append(xmlCollisionDefault);

        const xmlCollisionGeomDefault = xmlDoc.createElement("geom");
        xmlCollisionGeomDefault.setAttribute("group", "3");
        xmlCollisionDefault.append(xmlCollisionGeomDefault);

        const xmlGeneralDefault = xmlDoc.createElement("general");
        xmlGeneralDefault.setAttribute("dyntype", "none");
        xmlGeneralDefault.setAttribute("biastype", "affine");
        xmlDefaults.append(xmlGeneralDefault);

        return xmlDefaults;
    }

    _createXmlBody(xmlDoc, name, f, R, n, joint=null) {
        let pos = math.reshape(f.subset(math.index(math.range(0, 3), n)), [3]);
        const rot = (n > 0 ? math.reshape(R.subset(math.index(math.range(0, 3), math.range(0, 3), n-1)), [3, 3]) : null);
        const color = this.colors[n % this.colors.length];

        const xmlBody = xmlDoc.createElement("body");
        xmlBody.setAttribute("name", name);
        xmlBody.setAttribute("pos", pos.toArray().join(" "));

        if (rot != null) {
            const q = this._R2q(rot);
            xmlBody.setAttribute("quat", "" + q.w + " " + q.x + " " + q.y + " " + q.z);
        }

        if (joint != null) {
            const xmlJoint = xmlDoc.createElement("joint");
            xmlJoint.setAttribute("name", joint);
            xmlJoint.setAttribute("axis", "0 0 1");
            xmlJoint.setAttribute("armature", "0.1");
            xmlJoint.setAttribute("damping", "1");
            xmlJoint.setAttribute("stiffness", "10");
            xmlBody.append(xmlJoint);

            const radius = (n % 2) == 0 ? "0.028" : "0.03";

            let xmlGeom = xmlDoc.createElement("geom");
            xmlGeom.setAttribute("type", "cylinder");
            xmlGeom.setAttribute("size", radius + " 0.01");
            xmlGeom.setAttribute("mass", "1");
            xmlGeom.setAttribute("class", this.name + "_visual");
            xmlGeom.setAttribute("rgba", color);
            xmlBody.append(xmlGeom);

            xmlGeom = xmlDoc.createElement("geom");
            xmlGeom.setAttribute("type", "sphere");
            xmlGeom.setAttribute("size", "0.01");
            xmlGeom.setAttribute("class", this.name + "_collision");
            xmlBody.append(xmlGeom);
        } else {
            let xmlGeom = xmlDoc.createElement("geom");
            xmlGeom.setAttribute("type", "sphere");
            xmlGeom.setAttribute("size", "0.02");
            xmlGeom.setAttribute("mass", ".1");
            xmlGeom.setAttribute("class", this.name + "_visual");
            xmlGeom.setAttribute("rgba", color);
            xmlBody.append(xmlGeom);

            xmlGeom = xmlDoc.createElement("geom");
            xmlGeom.setAttribute("type", "sphere");
            xmlGeom.setAttribute("size", "0.02");
            xmlGeom.setAttribute("class", this.name + "_collision");
            xmlBody.append(xmlGeom);
        }

        if (n < this.q.length) {
            let p = math.reshape(f.subset(math.index(math.range(0, 3), n+1)), [3]).toArray();

            if (math.norm(p) > 1e-6) {
                let xmlGeom = xmlDoc.createElement("geom");
                xmlGeom.setAttribute("type", "cylinder");
                xmlGeom.setAttribute("size", "0.01");
                xmlGeom.setAttribute("fromto", "0 0 0 " + p.join(" "));
                xmlGeom.setAttribute("mass", "" + (1.0 * math.norm(p)));
                xmlGeom.setAttribute("class", this.name + "_visual");
                xmlGeom.setAttribute("rgba", color);
                xmlBody.append(xmlGeom);

                const p1 = math.multiply(math.divide(p, math.norm(p)), 0.01);
                const p2 = math.subtract(p, p1);

                xmlGeom = xmlDoc.createElement("geom");
                xmlGeom.setAttribute("type", "cylinder");
                xmlGeom.setAttribute("size", "0.01");
                xmlGeom.setAttribute("fromto", p1.join(" ") + " " + p2.join(" "));
                xmlGeom.setAttribute("class", this.name + "_collision");
                xmlBody.append(xmlGeom);
            }
        }

        return xmlBody;
    }

    _fkin(x) {
        const N = this.q.length;

        const x0 = [...x];
        x0.push(0);

        let Tf = math.identity(4);

        const R = math.zeros(3, 3, N);
        const f = math.zeros(3, N+1);

        for (let n = 0; n < N; ++n) {
            const ct = math.cos(x0[n] + this.q_offset[n]);
            const st = math.sin(x0[n] + this.q_offset[n]);
            const ca = math.cos(this.alpha[n]);
            const sa = math.sin(this.alpha[n]);

            Tf = math.matrix([[ct,    -st,     0,   this.r[n]   ],
                              [st*ca,  ct*ca, -sa, -this.d[n]*sa],
                              [st*sa,  ct*sa,  ca,  this.d[n]*ca],
                              [0,      0,      0,   1           ]]
            );

            R.subset(
                math.index(math.range(0, 3), math.range(0, 3), n),
                Tf.subset(math.index(math.range(0, 3), math.range(0, 3)))
            );

            f.subset(
                math.index(math.range(0, 3), n+1),
                Tf.subset(math.index(math.range(0, 3), 3))
            );
        }

        return [f, R];
    }

    _R2q(R) {
        const rot3x3 = new THREE.Matrix3().fromArray(math.flatten(math.transpose(R)).toArray());
        const rot4x4 = new THREE.Matrix4().setFromMatrix3(rot3x3);
        const q = new THREE.Quaternion().setFromRotationMatrix(rot4x4);
        return q;
    }

}

/*
 * SPDX-FileCopyrightText: Copyright © 2010-2024 three.js authors
 *
 * SPDX-FileContributor: three.js authors
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Customized version of the 'OutlinePass' class of three.js

Modifications includes using normal blending instead of additive (to have a solid color for the outline
instead of a transparent-like one) and a fix for transform controls (which are otherwise always displayed
with an outline with the 'hiddenEdgeColor').
*/
class OutlinePass extends Pass {

    constructor( resolution, scene, camera, transformControls, selectedObjects ) {

        super();

        this.renderScene = scene;
        this.renderCamera = camera;
        this.transformControls = transformControls;
        this.selectedObjects = selectedObjects !== undefined ? selectedObjects : [];
        this.visibleEdgeColor = new Color( 1, 1, 1 );
        this.hiddenEdgeColor = new Color( 0.1, 0.04, 0.02 );
        this.edgeGlow = 0.0;
        this.usePatternTexture = false;
        this.edgeThickness = 1.0;
        this.edgeStrength = 3.0;
        this.downSampleRatio = 2;
        this.pulsePeriod = 0;

        this._visibilityCache = new Map();
        this._selectionCache = new Set();

        this.resolution = ( resolution !== undefined ) ? new Vector2( resolution.x, resolution.y ) : new Vector2( 256, 256 );

        const resx = Math.round( this.resolution.x / this.downSampleRatio );
        const resy = Math.round( this.resolution.y / this.downSampleRatio );

        this.renderTargetMaskBuffer = new WebGLRenderTarget( this.resolution.x, this.resolution.y );
        this.renderTargetMaskBuffer.texture.name = 'OutlinePass.mask';
        this.renderTargetMaskBuffer.texture.generateMipmaps = false;

        this.depthMaterial = new MeshDepthMaterial();
        this.depthMaterial.side = DoubleSide;
        this.depthMaterial.depthPacking = RGBADepthPacking;
        this.depthMaterial.blending = NoBlending;

        this.prepareMaskMaterial = this.getPrepareMaskMaterial();
        this.prepareMaskMaterial.side = DoubleSide;
        this.prepareMaskMaterial.fragmentShader = replaceDepthToViewZ( this.prepareMaskMaterial.fragmentShader, this.renderCamera );

        this.renderTargetDepthBuffer = new WebGLRenderTarget( this.resolution.x, this.resolution.y, { type: HalfFloatType } );
        this.renderTargetDepthBuffer.texture.name = 'OutlinePass.depth';
        this.renderTargetDepthBuffer.texture.generateMipmaps = false;

        this.renderTargetMaskDownSampleBuffer = new WebGLRenderTarget( resx, resy, { type: HalfFloatType } );
        this.renderTargetMaskDownSampleBuffer.texture.name = 'OutlinePass.depthDownSample';
        this.renderTargetMaskDownSampleBuffer.texture.generateMipmaps = false;

        this.renderTargetBlurBuffer1 = new WebGLRenderTarget( resx, resy, { type: HalfFloatType } );
        this.renderTargetBlurBuffer1.texture.name = 'OutlinePass.blur1';
        this.renderTargetBlurBuffer1.texture.generateMipmaps = false;
        this.renderTargetBlurBuffer2 = new WebGLRenderTarget( Math.round( resx / 2 ), Math.round( resy / 2 ), { type: HalfFloatType } );
        this.renderTargetBlurBuffer2.texture.name = 'OutlinePass.blur2';
        this.renderTargetBlurBuffer2.texture.generateMipmaps = false;

        this.edgeDetectionMaterial = this.getEdgeDetectionMaterial();
        this.renderTargetEdgeBuffer1 = new WebGLRenderTarget( resx, resy, { type: HalfFloatType } );
        this.renderTargetEdgeBuffer1.texture.name = 'OutlinePass.edge1';
        this.renderTargetEdgeBuffer1.texture.generateMipmaps = false;
        this.renderTargetEdgeBuffer2 = new WebGLRenderTarget( Math.round( resx / 2 ), Math.round( resy / 2 ), { type: HalfFloatType } );
        this.renderTargetEdgeBuffer2.texture.name = 'OutlinePass.edge2';
        this.renderTargetEdgeBuffer2.texture.generateMipmaps = false;

        const MAX_EDGE_THICKNESS = 4;
        const MAX_EDGE_GLOW = 4;

        this.separableBlurMaterial1 = this.getSeperableBlurMaterial( MAX_EDGE_THICKNESS );
        this.separableBlurMaterial1.uniforms[ 'texSize' ].value.set( resx, resy );
        this.separableBlurMaterial1.uniforms[ 'kernelRadius' ].value = 1;
        this.separableBlurMaterial2 = this.getSeperableBlurMaterial( MAX_EDGE_GLOW );
        this.separableBlurMaterial2.uniforms[ 'texSize' ].value.set( Math.round( resx / 2 ), Math.round( resy / 2 ) );
        this.separableBlurMaterial2.uniforms[ 'kernelRadius' ].value = MAX_EDGE_GLOW;

        // Overlay material
        this.overlayMaterial = this.getOverlayMaterial();

        // copy material

        const copyShader = CopyShader;

        this.copyUniforms = UniformsUtils.clone( copyShader.uniforms );

        this.materialCopy = new ShaderMaterial( {
            uniforms: this.copyUniforms,
            vertexShader: copyShader.vertexShader,
            fragmentShader: copyShader.fragmentShader,
            blending: NoBlending,
            depthTest: false,
            depthWrite: false
        } );

        this.enabled = true;
        this.needsSwap = false;

        this._oldClearColor = new Color();
        this.oldClearAlpha = 1;

        this.fsQuad = new FullScreenQuad( null );

        this.tempPulseColor1 = new Color();
        this.tempPulseColor2 = new Color();
        this.textureMatrix = new Matrix4();

        function replaceDepthToViewZ( string, camera ) {

            const type = camera.isPerspectiveCamera ? 'perspective' : 'orthographic';

            return string.replace( /DEPTH_TO_VIEW_Z/g, type + 'DepthToViewZ' );

        }

    }

    dispose() {

        this.renderTargetMaskBuffer.dispose();
        this.renderTargetDepthBuffer.dispose();
        this.renderTargetMaskDownSampleBuffer.dispose();
        this.renderTargetBlurBuffer1.dispose();
        this.renderTargetBlurBuffer2.dispose();
        this.renderTargetEdgeBuffer1.dispose();
        this.renderTargetEdgeBuffer2.dispose();

        this.depthMaterial.dispose();
        this.prepareMaskMaterial.dispose();
        this.edgeDetectionMaterial.dispose();
        this.separableBlurMaterial1.dispose();
        this.separableBlurMaterial2.dispose();
        this.overlayMaterial.dispose();
        this.materialCopy.dispose();

        this.fsQuad.dispose();

    }

    setSize( width, height ) {

        this.renderTargetMaskBuffer.setSize( width, height );
        this.renderTargetDepthBuffer.setSize( width, height );

        let resx = Math.round( width / this.downSampleRatio );
        let resy = Math.round( height / this.downSampleRatio );
        this.renderTargetMaskDownSampleBuffer.setSize( resx, resy );
        this.renderTargetBlurBuffer1.setSize( resx, resy );
        this.renderTargetEdgeBuffer1.setSize( resx, resy );
        this.separableBlurMaterial1.uniforms[ 'texSize' ].value.set( resx, resy );

        resx = Math.round( resx / 2 );
        resy = Math.round( resy / 2 );

        this.renderTargetBlurBuffer2.setSize( resx, resy );
        this.renderTargetEdgeBuffer2.setSize( resx, resy );

        this.separableBlurMaterial2.uniforms[ 'texSize' ].value.set( resx, resy );

    }

    updateSelectionCache() {

        const cache = this._selectionCache;

        function gatherSelectedMeshesCallBack( object ) {

            if ( object.isMesh ) cache.add( object );

        }

        cache.clear();

        for ( let i = 0; i < this.selectedObjects.length; i ++ ) {

            const selectedObject = this.selectedObjects[ i ];
            selectedObject.traverse( gatherSelectedMeshesCallBack );

        }

    }

    changeVisibilityOfSelectedObjects( bVisible ) {

        const cache = this._visibilityCache;

        for ( const mesh of this._selectionCache ) {

            if ( bVisible === true ) {

                mesh.visible = cache.get( mesh );

            } else {

                cache.set( mesh, mesh.visible );
                mesh.visible = bVisible;

            }

        }

    }

    changeVisibilityOfNonSelectedObjects( bVisible ) {

        const visibilityCache = this._visibilityCache;
        const selectionCache = this._selectionCache;

        function VisibilityChangeCallBack( object ) {

            if ( object.isMesh || object.isSprite ) {

                // only meshes and sprites are supported by OutlinePass

                if ( ! selectionCache.has( object ) ) {

                    const visibility = object.visible;

                    if ( bVisible === false || visibilityCache.get( object ) === true ) {

                        object.visible = bVisible;

                    }

                    visibilityCache.set( object, visibility );

                }

            } else if ( object.isPoints || object.isLine ) {

                // the visibilty of points and lines is always set to false in order to
                // not affect the outline computation

                if ( bVisible === true ) {

                    object.visible = visibilityCache.get( object ); // restore

                } else {

                    visibilityCache.set( object, object.visible );
                    object.visible = bVisible;

                }

            }

        }

        this.renderScene.traverse( VisibilityChangeCallBack );

    }

    updateTextureMatrix() {

        this.textureMatrix.set( 0.5, 0.0, 0.0, 0.5,
            0.0, 0.5, 0.0, 0.5,
            0.0, 0.0, 0.5, 0.5,
            0.0, 0.0, 0.0, 1.0 );
        this.textureMatrix.multiply( this.renderCamera.projectionMatrix );
        this.textureMatrix.multiply( this.renderCamera.matrixWorldInverse );

    }

    render( renderer, writeBuffer, readBuffer, deltaTime, maskActive ) {

        if ( this.selectedObjects.length > 0 ) {
            this.renderCamera.layers.enableAll();

            renderer.getClearColor( this._oldClearColor );
            this.oldClearAlpha = renderer.getClearAlpha();
            const oldAutoClear = renderer.autoClear;

            renderer.autoClear = false;

            if ( maskActive ) renderer.state.buffers.stencil.setTest( false );

            renderer.setClearColor( 0xffffff, 1 );

            this.updateSelectionCache();

            // Make selected objects invisible
            this.changeVisibilityOfSelectedObjects( false );

            const currentBackground = this.renderScene.background;
            this.renderScene.background = null;

            // 1. Draw Non Selected objects in the depth buffer
            this.renderScene.overrideMaterial = this.depthMaterial;
            renderer.setRenderTarget( this.renderTargetDepthBuffer );
            renderer.clear();
            renderer.render( this.renderScene, this.renderCamera );

            // Make selected objects visible
            this.changeVisibilityOfSelectedObjects( true );
            this._visibilityCache.clear();

            // Update Texture Matrix for Depth compare
            this.updateTextureMatrix();

            // Make non selected objects invisible, and draw only the selected objects, by comparing the depth buffer of non selected objects
            const selectionCache = new Map();

            function VisibilityChangeCallBack( object ) {
                selectionCache.set(object, object.visible);
                object.visible = false;
            }

            this.transformControls.transformControls.getHelper().traverse( VisibilityChangeCallBack );

            this.changeVisibilityOfNonSelectedObjects( false );
            this.renderScene.overrideMaterial = this.prepareMaskMaterial;
            this.prepareMaskMaterial.uniforms[ 'cameraNearFar' ].value.set( this.renderCamera.near, this.renderCamera.far );
            this.prepareMaskMaterial.uniforms[ 'depthTexture' ].value = this.renderTargetDepthBuffer.texture;
            this.prepareMaskMaterial.uniforms[ 'textureMatrix' ].value = this.textureMatrix;
            renderer.setRenderTarget( this.renderTargetMaskBuffer );
            renderer.clear();
            renderer.render( this.renderScene, this.renderCamera );
            this.renderScene.overrideMaterial = null;
            this.changeVisibilityOfNonSelectedObjects( true );
            this._visibilityCache.clear();
            this._selectionCache.clear();

            function VisibilityChangeCallBack2( object ) {
                object.visible = selectionCache.get(object);
            }

            this.transformControls.transformControls._root.traverse( VisibilityChangeCallBack2 );


            this.renderScene.background = currentBackground;

            // 2. Downsample to Half resolution
            this.fsQuad.material = this.materialCopy;
            this.copyUniforms[ 'tDiffuse' ].value = this.renderTargetMaskBuffer.texture;
            renderer.setRenderTarget( this.renderTargetMaskDownSampleBuffer );
            renderer.clear();
            this.fsQuad.render( renderer );

            this.tempPulseColor1.copy( this.visibleEdgeColor );
            this.tempPulseColor2.copy( this.hiddenEdgeColor );

            if ( this.pulsePeriod > 0 ) {

                const scalar = ( 1 + 0.25 ) / 2 + Math.cos( performance.now() * 0.01 / this.pulsePeriod ) * ( 1.0 - 0.25 ) / 2;
                this.tempPulseColor1.multiplyScalar( scalar );
                this.tempPulseColor2.multiplyScalar( scalar );

            }

            // 3. Apply Edge Detection Pass
            this.fsQuad.material = this.edgeDetectionMaterial;
            this.edgeDetectionMaterial.uniforms[ 'maskTexture' ].value = this.renderTargetMaskDownSampleBuffer.texture;
            this.edgeDetectionMaterial.uniforms[ 'texSize' ].value.set( this.renderTargetMaskDownSampleBuffer.width, this.renderTargetMaskDownSampleBuffer.height );
            this.edgeDetectionMaterial.uniforms[ 'visibleEdgeColor' ].value = this.tempPulseColor1;
            this.edgeDetectionMaterial.uniforms[ 'hiddenEdgeColor' ].value = this.tempPulseColor2;
            renderer.setRenderTarget( this.renderTargetEdgeBuffer1 );
            renderer.clear();
            this.fsQuad.render( renderer );

            // 4. Apply Blur on Half res
            this.fsQuad.material = this.separableBlurMaterial1;
            this.separableBlurMaterial1.uniforms[ 'colorTexture' ].value = this.renderTargetEdgeBuffer1.texture;
            this.separableBlurMaterial1.uniforms[ 'direction' ].value = OutlinePass.BlurDirectionX;
            this.separableBlurMaterial1.uniforms[ 'kernelRadius' ].value = this.edgeThickness;
            renderer.setRenderTarget( this.renderTargetBlurBuffer1 );
            renderer.clear();
            this.fsQuad.render( renderer );
            this.separableBlurMaterial1.uniforms[ 'colorTexture' ].value = this.renderTargetBlurBuffer1.texture;
            this.separableBlurMaterial1.uniforms[ 'direction' ].value = OutlinePass.BlurDirectionY;
            renderer.setRenderTarget( this.renderTargetEdgeBuffer1 );
            renderer.clear();
            this.fsQuad.render( renderer );

            // Apply Blur on quarter res
            this.fsQuad.material = this.separableBlurMaterial2;
            this.separableBlurMaterial2.uniforms[ 'colorTexture' ].value = this.renderTargetEdgeBuffer1.texture;
            this.separableBlurMaterial2.uniforms[ 'direction' ].value = OutlinePass.BlurDirectionX;
            renderer.setRenderTarget( this.renderTargetBlurBuffer2 );
            renderer.clear();
            this.fsQuad.render( renderer );
            this.separableBlurMaterial2.uniforms[ 'colorTexture' ].value = this.renderTargetBlurBuffer2.texture;
            this.separableBlurMaterial2.uniforms[ 'direction' ].value = OutlinePass.BlurDirectionY;
            renderer.setRenderTarget( this.renderTargetEdgeBuffer2 );
            renderer.clear();
            this.fsQuad.render( renderer );

            // Blend it additively over the input texture
            this.fsQuad.material = this.overlayMaterial;
            this.overlayMaterial.uniforms[ 'maskTexture' ].value = this.renderTargetMaskBuffer.texture;
            this.overlayMaterial.uniforms[ 'edgeTexture1' ].value = this.renderTargetEdgeBuffer1.texture;
            this.overlayMaterial.uniforms[ 'edgeTexture2' ].value = this.renderTargetEdgeBuffer2.texture;
            this.overlayMaterial.uniforms[ 'patternTexture' ].value = this.patternTexture;
            this.overlayMaterial.uniforms[ 'edgeStrength' ].value = this.edgeStrength;
            this.overlayMaterial.uniforms[ 'edgeGlow' ].value = this.edgeGlow;
            this.overlayMaterial.uniforms[ 'usePatternTexture' ].value = this.usePatternTexture;


            if ( maskActive ) renderer.state.buffers.stencil.setTest( true );

            renderer.setRenderTarget( readBuffer );
            this.fsQuad.render( renderer );

            renderer.setClearColor( this._oldClearColor, this.oldClearAlpha );
            renderer.autoClear = oldAutoClear;

            this.renderCamera.layers.disableAll();
            this.renderCamera.layers.enable(0);
        }

        if ( this.renderToScreen ) {

            this.fsQuad.material = this.materialCopy;
            this.copyUniforms[ 'tDiffuse' ].value = readBuffer.texture;
            renderer.setRenderTarget( null );
            this.fsQuad.render( renderer );

        }

    }

    getPrepareMaskMaterial() {

        return new ShaderMaterial( {

            uniforms: {
                'depthTexture': { value: null },
                'cameraNearFar': { value: new Vector2( 0.5, 0.5 ) },
                'textureMatrix': { value: null }
            },

            vertexShader:
                `#include <morphtarget_pars_vertex>
                #include <skinning_pars_vertex>

                varying vec4 projTexCoord;
                varying vec4 vPosition;
                uniform mat4 textureMatrix;

                void main() {

                    #include <skinbase_vertex>
                    #include <begin_vertex>
                    #include <morphtarget_vertex>
                    #include <skinning_vertex>
                    #include <project_vertex>

                    vPosition = mvPosition;

                    vec4 worldPosition = vec4( transformed, 1.0 );

                    #ifdef USE_INSTANCING

                        worldPosition = instanceMatrix * worldPosition;

                    #endif

                    worldPosition = modelMatrix * worldPosition;

                    projTexCoord = textureMatrix * worldPosition;

                }`,

            fragmentShader:
                `#include <packing>
                varying vec4 vPosition;
                varying vec4 projTexCoord;
                uniform sampler2D depthTexture;
                uniform vec2 cameraNearFar;

                void main() {

                    float depth = unpackRGBAToDepth(texture2DProj( depthTexture, projTexCoord ));
                    float viewZ = - DEPTH_TO_VIEW_Z( depth, cameraNearFar.x, cameraNearFar.y );
                    float depthTest = (-vPosition.z > viewZ) ? 1.0 : 0.0;
                    gl_FragColor = vec4(0.0, depthTest, 1.0, 1.0);

                }`

        } );

    }

    getEdgeDetectionMaterial() {

        return new ShaderMaterial( {

            uniforms: {
                'maskTexture': { value: null },
                'texSize': { value: new Vector2( 0.5, 0.5 ) },
                'visibleEdgeColor': { value: new Vector3( 1.0, 1.0, 1.0 ) },
                'hiddenEdgeColor': { value: new Vector3( 1.0, 1.0, 1.0 ) },
            },

            vertexShader:
                `varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                }`,

            fragmentShader:
                `varying vec2 vUv;

                uniform sampler2D maskTexture;
                uniform vec2 texSize;
                uniform vec3 visibleEdgeColor;
                uniform vec3 hiddenEdgeColor;

                void main() {
                    vec2 invSize = 1.0 / texSize;
                    vec4 uvOffset = vec4(1.0, 0.0, 0.0, 1.0) * vec4(invSize, invSize);
                    vec4 c1 = texture2D( maskTexture, vUv + uvOffset.xy);
                    vec4 c2 = texture2D( maskTexture, vUv - uvOffset.xy);
                    vec4 c3 = texture2D( maskTexture, vUv + uvOffset.yw);
                    vec4 c4 = texture2D( maskTexture, vUv - uvOffset.yw);
                    float diff1 = (c1.r - c2.r)*0.5;
                    float diff2 = (c3.r - c4.r)*0.5;
                    float d = length( vec2(diff1, diff2) );
                    float a1 = min(c1.g, c2.g);
                    float a2 = min(c3.g, c4.g);
                    float visibilityFactor = min(a1, a2);
                    vec3 edgeColor = 1.0 - visibilityFactor > 0.001 ? visibleEdgeColor : hiddenEdgeColor;
                    gl_FragColor = vec4(edgeColor, 1.0) * vec4(d);
                }`
        } );

    }

    getSeperableBlurMaterial( maxRadius ) {

        return new ShaderMaterial( {

            defines: {
                'MAX_RADIUS': maxRadius,
            },

            uniforms: {
                'colorTexture': { value: null },
                'texSize': { value: new Vector2( 0.5, 0.5 ) },
                'direction': { value: new Vector2( 0.5, 0.5 ) },
                'kernelRadius': { value: 1.0 }
            },

            vertexShader:
                `varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                }`,

            fragmentShader:
                `#include <common>
                varying vec2 vUv;
                uniform sampler2D colorTexture;
                uniform vec2 texSize;
                uniform vec2 direction;
                uniform float kernelRadius;

                float gaussianPdf(in float x, in float sigma) {
                    return 0.39894 * exp( -0.5 * x * x/( sigma * sigma))/sigma;
                }

                void main() {
                    vec2 invSize = 1.0 / texSize;
                    float sigma = kernelRadius/2.0;
                    float weightSum = gaussianPdf(0.0, sigma);
                    vec4 diffuseSum = texture2D( colorTexture, vUv) * weightSum;
                    vec2 delta = direction * invSize * kernelRadius/float(MAX_RADIUS);
                    vec2 uvOffset = delta;
                    for( int i = 1; i <= MAX_RADIUS; i ++ ) {
                        float x = kernelRadius * float(i) / float(MAX_RADIUS);
                        float w = gaussianPdf(x, sigma);
                        vec4 sample1 = texture2D( colorTexture, vUv + uvOffset);
                        vec4 sample2 = texture2D( colorTexture, vUv - uvOffset);
                        diffuseSum += ((sample1 + sample2) * w);
                        weightSum += (2.0 * w);
                        uvOffset += delta;
                    }
                    gl_FragColor = diffuseSum/weightSum;
                }`
        } );

    }

    getOverlayMaterial() {

        return new ShaderMaterial( {

            uniforms: {
                'maskTexture': { value: null },
                'edgeTexture1': { value: null },
                'edgeTexture2': { value: null },
                'patternTexture': { value: null },
                'edgeStrength': { value: 1.0 },
                'edgeGlow': { value: 1.0 },
                'usePatternTexture': { value: 0.0 }
            },

            vertexShader:
                `varying vec2 vUv;

                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
                }`,

            fragmentShader:
                `varying vec2 vUv;

                uniform sampler2D maskTexture;
                uniform sampler2D edgeTexture1;
                uniform sampler2D edgeTexture2;
                uniform sampler2D patternTexture;
                uniform float edgeStrength;
                uniform float edgeGlow;
                uniform bool usePatternTexture;

                void main() {
                    vec4 edgeValue1 = texture2D(edgeTexture1, vUv);
                    vec4 edgeValue2 = texture2D(edgeTexture2, vUv);
                    vec4 maskColor = texture2D(maskTexture, vUv);
                    vec4 patternColor = texture2D(patternTexture, 6.0 * vUv);
                    float visibilityFactor = 1.0 - maskColor.g > 0.0 ? 1.0 : 0.5;
                    vec4 edgeValue = edgeValue1 + edgeValue2 * edgeGlow;
                    vec4 finalColor = edgeStrength * maskColor.r * edgeValue;
                    if(usePatternTexture)
                        finalColor += + visibilityFactor * (1.0 - maskColor.r) * (1.0 - patternColor.r);
                    gl_FragColor = finalColor;
                }`,
            blending: NormalBlending,
            depthTest: false,
            depthWrite: false,
            transparent: true
        } );

    }

}

OutlinePass.BlurDirectionX = new Vector2( 1.0, 0.0 );
OutlinePass.BlurDirectionY = new Vector2( 0.0, 1.0 );

/*
 * SPDX-FileCopyrightText: Copyright © 2010-2024 three.js authors
 *
 * SPDX-FileContributor: three.js authors
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Customized version of the 'RenderPass' class of three.js

This version only renders a specific layer.
*/
class LayerRenderPass extends Pass {

    constructor( scene, camera, layer = 0, overrideMaterial = null, clearColor = null, clearAlpha = null ) {
        super();

        this.scene = scene;
        this.camera = camera;
        this.layer = layer;

        this.overrideMaterial = overrideMaterial;

        this.clearColor = clearColor;
        this.clearAlpha = clearAlpha;

        this.clear = false;
        this.clearDepth = false;
        this.needsSwap = false;
        this._oldClearColor = new Color();
    }

    render( renderer, writeBuffer, readBuffer /*, deltaTime, maskActive */ ) {
        const oldAutoClear = renderer.autoClear;
        renderer.autoClear = false;

        let oldClearAlpha, oldOverrideMaterial;

        if ( this.overrideMaterial !== null ) {
            oldOverrideMaterial = this.scene.overrideMaterial;
            this.scene.overrideMaterial = this.overrideMaterial;
        }

        if ( this.clearColor !== null ) {
            renderer.getClearColor( this._oldClearColor );
            renderer.setClearColor( this.clearColor, renderer.getClearAlpha() );
        }

        if ( this.clearAlpha !== null ) {
            oldClearAlpha = renderer.getClearAlpha();
            renderer.setClearAlpha( this.clearAlpha );
        }

        if ( this.clearDepth == true ) {
            renderer.clearDepth();
        }

        renderer.setRenderTarget( this.renderToScreen ? null : readBuffer );

        if ( this.clear === true ) {
            // TODO: Avoid using autoClear properties, see https://github.com/mrdoob/three.js/pull/15571#issuecomment-465669600
            renderer.clear( renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil );
        }

        this.camera.layers.disableAll();
        this.camera.layers.enable(this.layer);

        renderer.render( this.scene, this.camera );

        this.camera.layers.disable(this.layer);
        this.camera.layers.enable(0);

        // restore

        if ( this.clearColor !== null ) {
            renderer.setClearColor( this._oldClearColor );
        }

        if ( this.clearAlpha !== null ) {
            renderer.setClearAlpha( oldClearAlpha );
        }

        if ( this.overrideMaterial !== null ) {
            this.scene.overrideMaterial = oldOverrideMaterial;
        }

        renderer.autoClear = oldAutoClear;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



const InteractionStates = Object.freeze({
    Default: Symbol("default"),
    Manipulation: Symbol("manipulation"),
    JointHovering: Symbol("joint_hovering"),
    JointDisplacement: Symbol("joint_displacement"),
    LinkDisplacement: Symbol("link_displacement"),
});


const Passes = Object.freeze({
    BaseRenderPass: Symbol(0),
    NoShadowsRenderPass: Symbol(1),
    TopRenderPass: Symbol(2),
    OutputPass: Symbol(3),
});


const Layers = Object.freeze({
    Base: 0,
    NoShadows: 1,
    Top: 2,
    User: 3,
    Labels: 31,
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

    If no DOM element is provided, one is created. It is the duty of the caller to insert it
    somewhere in the DOM (see 'Viewer3D.domElement').


    Optional parameters:
        shadows (bool):
            enable the rendering of the shadows (default: true)

        show_joint_positions (bool):
            enable the display of an visual indicator around each joint (default: false)

        joint_position_colors (list):
            the colors to use for the visual indicators around the joints (see
            "show_joint_positions", default: all 0xff0000)

        show_axes (bool):
            enable the display of the world coordinates axes (default: false)

        statistics (bool):
            enable the display of statistics about the rendering performance (default: false)

        external_loop (bool):
            indicates that the rendering frequency is controlled by the user application
            (default: false)


    Composition:
        This is an advanced topic that is only relevant if you need to customize the rendering
        beyond the standard features of viewer3d.js.

        3D objects can be put on different layers, each rendered separately. Additionaly,
        the rendering is based on three.js' EffectComposer, allowing the user to apply additional
        effects as needed (by adding/removing render passes).

        4 layers are pre-defined:

            - Layers.Base (0): most objects should be put in that layer. This is the only layer
                               in which objects are casting shadows.
            - Layers.NoShadows (1): Objects that should not cast shadows should be put in that
                                    layer
            - Layers.Top (2): objects that should be rendered after clearing the depth buffer
                              (= "on top of everything else") should be put in that layer
            - Layers.Labels (31): texts to be rendered in the 3D world must be in that layer

        The user can add its own layers (starting at index 'Layers.User'), but will also need
        to also add corresponding render passes.

        The default passes used are:

            - LayerRenderPass: render objects in layer 'Layers.Base' (id: Passes.BaseRenderPass)
            - LayerRenderPass: render objects in layer 'Layers.NoShadows'
                                 (id: Passes.NoShadowsRenderPass)
            - LayerRenderPass: render objects in layer 'Layers.Top'
                                 (id: Passes.TopRenderPass). Note that the depth buffer is cleared
                                 by this pass.
            - OutputPass: tone mapping, sRGB conversion (id: Passes.OutputPass)

        Note that transparent objects might be rendered with incorrect colors due to an issue
        in three.js. You might have to adjust their colors and/or opacity.
    */
    constructor(domElement, parameters) {
        this.parameters = this._checkParameters(parameters);

        this.domElement = domElement;

        if (this.domElement == undefined)
            this.domElement = document.createElement('div');

        if (!this.domElement.classList.contains('viewer3d'))
            this.domElement.classList.add('viewer3d');

        this.camera = null;
        this.scene = null;
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
        this.gaussians = new ObjectList();

        this.interactionState = InteractionStates.Default;

        this.hoveredRobot = null;
        this.hoveredGroup = null;
        this.hoveredJoint = null;
        this.previousPointer = null;
        this.didClick = false;

        this.planarIkControls = new PlanarIKControls();
        this.kinematicChain = null;

        this.toolbar = null;

        this.renderer = null;
        this.labelRenderer = null;

        this.composer = null;
        this.passes = null;

        this.clock = new THREE.Clock();

        this.cameraControl = null;
        this.transformControls = null;
        this.stats = null;

        this.raycaster = new THREE.Raycaster();
        this.raycaster.layers.enableAll();

        this.logmap = null;

        this.renderingCallback = null;
        this.renderingCallbackTime = null;
        this.renderingCallbackTimestep = -1.0;

        this.controlsEnabled = true;
        this.endEffectorManipulationEnabled = true;
        this.jointsManipulationEnabled = true;
        this.linksManipulationEnabled = true;
        this.objectsManipulationEnabled = true;
        this.forceImpulsesEnabled = false;
        this.toolsEnabled = true;

        this.forceImpulses = 0.0;

        this.controlStartedCallback = null;
        this.controlEndedCallback = null;

        this.mustStop = false;

        this._initScene();

        if (!this.parameters.get('external_loop'))
            this.render();
    }


    dispose() {
        this.renderer.dispose();
    }


    /* Register a function that should be called once per frame.

    This callback function can for example be used to update the positions of the joints.

    The signature of the callback function is: callback(delta), with 'delta' the time elapsed
    since the last frame, in seconds.

    Note that only one function can be registered at a time. If 'callback' is 'null', no
    function is called anymore.
    */
    setRenderingCallback(renderingCallback, timestep=-1.0) {
        this.renderingCallback = renderingCallback;
        this.renderingCallbackTimestep = timestep;
        this.renderingCallbackTime = null;
    }


    setControlCallbacks(startCallback, endCallback) {
        this.controlStartedCallback = startCallback;
        this.controlEndedCallback = endCallback;
    }


    /* Enables/disables the manipulation controls

    Manipulation controls include the end-effector and the target manipulators.
    */
    enableControls(enabled) {
        this.controlsEnabled = enabled;
        this.transformControls.enable(enabled);
        this.enableRobotTools(this.toolsEnabled);
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
        this.endEffectorManipulationEnabled = enabled && (this.planarIkControls != null);

        if (enabled) {
            for (let name in self.robots) {
                const robot = self.robots[name];
                if (robot.controlsEnabled && robot.tools[0].tcpTarget == null)
                    robot._createTcpTargets();
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
        this.linksManipulationEnabled = enabled && (this.planarIkControls != null);

        if ((this.interactionState == InteractionStates.JointHovering) ||
            (this.interactionState == InteractionStates.JointDisplacement) ||
            (this.interactionState == InteractionStates.LinkDisplacement)) {
                this._switchToInteractionState(InteractionStates.Default);
        }

        if (enabled)
            this.forceImpulsesEnabled = false;
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


    /* Enables/disables the manipulation of the objects (like the gaussians)

    Note that if 'Viewer3D.controlsEnabled' is 'false', the transforms of the objects
    can't be changed using the mouse regardless of the value of this property.
    */
    enableObjectsManipulation(enabled) {
        this.objectsManipulationEnabled = enabled;
    }


    enableForceImpulses(enabled, amount=0.0) {
        this.forceImpulsesEnabled = enabled && (amount > 0.0);
        this.forceImpulses = amount;

        if ((this.interactionState == InteractionStates.JointHovering) ||
            (this.interactionState == InteractionStates.JointDisplacement) ||
            (this.interactionState == InteractionStates.LinkDisplacement)) {
                this._switchToInteractionState(InteractionStates.Default);
        }

        if (enabled)
            this.linksManipulationEnabled = false;
    }


    areForceImpulsesEnabled() {
        return this.forceImpulsesEnabled;
    }


    /* Indicates if the manipulation of the objects (like the gaussians) is enabled

    Note that if 'Viewer3D.controlsEnabled' is 'false', the transforms of the objects
    can't be changed using the mouse regardless of the value of this property.
    */
    isObjectsManipulationEnabled() {
        return this.objectsManipulationEnabled;
    }


    enableRobotTools(enabled) {
        this.toolsEnabled = enabled;

        for (const name in this.robots)
            this.robots[name]._enableTools(this.toolsEnabled && this.controlsEnabled && robot.controlsEnabled);
    }


    areRobotToolsEnabled() {
        return this.toolsEnabled;
    }


    /* Change the layer in which new objects are created

    Parameters:
        layer (int): Index of the layer
    */
    activateLayer(layer) {
        this.activeLayer = layer;

        this.passes[Passes.NoShadowsRenderPass].enabled = true;
        this.passes[Passes.TopRenderPass].enabled = true;
    }


    addPassBefore(pass, standardPassId) {
        let index = this.composer.passes.indexOf(this.passes[standardPassId]);
        this.composer.insertPass(pass, index);
    }


    addPassAfter(pass, standardPassId) {
        let index = this.composer.passes.indexOf(this.passes[standardPassId]);
        this.composer.insertPass(pass, index+1);
    }


    loadScene(filename, robotBuilders=null) {
        if (this.physicsSimulator != null) {
            for (const name in this.robots)
                this.robots[name].destroy();

            const root = this.physicsSimulator.root;
            root.parent.remove(root);

            this.physicsSimulator.destroy();
            this.physicsSimulator = null;

            this.robots = {};
            this.skyboxScene = null;
            this.scene.fog = null;

            if (this.haze != null) {
                this.haze.parent.remove(this.haze);
                this.haze = null;
            }

            this.transformControls.detach();
        }

        this.physicsSimulator = loadScene(filename, robotBuilders);

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
        this.renderingCallbackTime = null;

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
            -Math.sin(camera.azimuth * Math.PI / 180.0),
            Math.sin(-camera.elevation * Math.PI / 180.0),
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


    createRobot(name, configuration, prefix=null, parameters=null) {
        if (this.physicsSimulator == null)
            return null;

        if (name in this.robots)
            return null;

        const robot = this.physicsSimulator.createRobot(name, configuration, prefix);
        if (robot == null)
            return;

        const robotParameters = this._checkRobotParameters(parameters);

        robot.controlsEnabled = robotParameters.get('controlsEnabled');

        const toolEnabled = this.toolsEnabled && this.controlsEnabled && robot.controlsEnabled;

        const robotsWithControlsEnabled = Object.values(this.robots).filter((r) => r.controlsEnabled);

        if (toolEnabled && (robotsWithControlsEnabled.length == 1))
        {
            for (const name in this.robots) {
                this.robots[name]._enableTools(false);
                this.robots[name]._enableTools(true);
            }
        }

        if (toolEnabled && (robotsWithControlsEnabled.length == 0) && (configuration.tools.length == 1) &&
            (configuration.tools[0].root != null)) {
            robot.enableTool(toolEnabled, new GripperToolbarSection(this.toolbar, robot));
        } else {
            robot._enableTools(toolEnabled);
        }

        this.physicsSimulator.simulation.forward();
        this.physicsSimulator.synchronize();

        robot.layers.disableAll();
        robot.layers.enable(this.activeLayer);

        this.robots[name] = robot;

        if (this.parameters.get('show_joint_positions')) {
            robot.createJointPositionHelpers(
                this.scene, Layers.NoShadows, this.parameters.get('joint_position_colors')
            );

            this.passes[Passes.NoShadowsRenderPass].enabled = true;
            this.passes[Passes.TopRenderPass].enabled = true;
        }

        if (robotParameters.get('color') != null) {
            let color = robotParameters.get('color');
            color = new THREE.Color().setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace);

            for (let segment of robot.segments)
                segment.visual.meshes.forEach((mesh) => modifyMaterialColor(mesh, color));

            for (let tool of robot.tools)
                tool.visual.meshes.forEach((mesh) => modifyMaterialColor(mesh, color));
        }

        if (robotParameters.get('use_toon_shader')) {
            for (let segment of robot.segments)
                segment.visual.meshes.forEach((mesh) => enableToonShading(mesh));

            for (let tool of robot.tools)
                tool.visual.meshes.forEach((mesh) => enableToonShading(mesh));

        } else if (robotParameters.get('use_light_toon_shader')) {
            for (let segment of robot.segments)
                segment.visual.meshes.forEach((mesh) => enableLightToonShading(mesh));

            for (let tool of robot.tools)
                tool.visual.meshes.forEach((mesh) => enableLightToonShading(mesh));
        }

        if (this.endEffectorManipulationEnabled && robot.controlsEnabled)
            robot._createTcpTarget();

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
        listener (function): Function to call when the target is moved/rotated using the mouse
        parameters (dict): Additional shape-dependent parameters (radius, width, height, ...) and opacity
    */
    addTarget(name, position, orientation, color, shape=Shapes.Cube, listener=null, parameters=null) {
        const target = this.targets.create(name, position, orientation, color, shape, listener, parameters);

        target.layers.disableAll();
        target.layers.enable(this.activeLayer);

        this.scene.add(target);
        return target;
    }


    /* Remove a target from the scene.

    Parameters:
        name (str): Name of the target
    */
    removeTarget(name) {
        const target = this.targets.get(name);

        if (this.transformControls.getAttachedObject() == target)
            this.transformControls.detach();

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

        arrow.layers.disableAll();
        arrow.layers.enable(this.activeLayer);

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

        path.layers.disableAll();
        path.layers.enable(this.activeLayer);

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

        point.layers.disableAll();
        point.layers.enable(this.activeLayer);

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


    /* Add a gaussian to the scene

    Parameters:
        name (str): Name of the gaussian
        mu (Vector3): Position of the gaussian
        sigma (Matrix): Covariance matrix of the gaussian
        color (int/str): Color of the gaussian (by default: 0xffff00)
        listener (function): Function to call when the gaussian is modified using the mouse
    */
    addGaussian(name, mu, sigma, color=0xffff00, listener=null) {
        const gaussian = new Gaussian(name, mu, sigma, color, listener);

        gaussian.layers.disableAll();
        gaussian.layers.enable(this.activeLayer);

        this.gaussians.add(gaussian);
        this.scene.add(gaussian);
        return gaussian;
    }


    /* Remove a gaussian from the scene.

    Parameters:
        name (str): Name of the gaussian
    */
    removeGaussian(name) {
        const gaussian = this.gaussians.get(name);

        if (this.transformControls.getAttachedObject() == gaussian)
            this.transformControls.detach();

        this.gaussians.destroy(name);
    }


    /* Returns a gaussian from the scene.

    Parameters:
        name (str): Name of the gaussian
    */
    getGaussian(name) {
        return this.gaussians.get(name);
    }


    getPhysicalBody(name) {
        const bodyId = this.physicsSimulator.getBodyId(name);
        if (bodyId == null)
            return null;

        return new PhysicalBody(name, bodyId, this.physicsSimulator);
    }


    translateCamera(delta) {
        this.cameraControl.target.add(delta);
        this.cameraControl.update();
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


    async stop() {
        if (this.parameters.get('external_loop'))
            return;

        this.mustStop = true;

        const viewer = this;

        function callback(resolve) {
            if (!viewer.mustStop)
                resolve();
            else
                setTimeout(() => { callback(resolve); });
        }
        const promise = new Promise((resolve, reject) => {
            setTimeout(() => {
                callback(resolve);
            }, 10);
        });

        await promise;
    }


    _checkParameters(parameters) {
        if (parameters == null)
            parameters = new Map();
        else if (!(parameters instanceof Map))
            parameters = new Map(Object.entries(parameters));

        const defaults = new Map([
            ['joint_position_colors', []],
            ['joint_position_layer', null],
            ['shadows', true],
            ['show_joint_positions', false],
            ['show_axes', false],
            ['statistics', false],
            ['external_loop', false],
        ]);

        return new Map([...defaults, ...parameters]);
    }


    _checkRobotParameters(parameters) {
        if (parameters == null)
            parameters = new Map();
        else if (!(parameters instanceof Map))
            parameters = new Map(Object.entries(parameters));

        const defaults = new Map([
            ['use_toon_shader', false],
            ['use_light_toon_shader', false],
            ['hue', null],
            ['color', null],
            ['controlsEnabled', true],
        ]);

        return new Map([...defaults, ...parameters]);
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

        THREE.Object3D.DefaultUp = new THREE.Vector3(0, 0, 1);

        // Camera
        this.camera = new THREE.PerspectiveCamera(45, this.domElement.clientWidth / this.domElement.clientHeight, 0.1, 50);
        this.camera.up.set(0, 0, 1);
        this.camera.position.set(1, 1, -2);
        this.camera.layers.disableAll();
        this.camera.layers.enable(0);

        this.scene = new THREE.Scene();

        if (this.parameters.get('show_axes'))
        {
            let material = new THREE.LineBasicMaterial({
                color: 0xFF0000,
            });

            let points = [];
            points.push( new THREE.Vector3(0, 0, 0) );
            points.push( new THREE.Vector3(1, 0, 0) );

            let geometry = new THREE.BufferGeometry().setFromPoints(points);
            let line = new THREE.Line(geometry, material);
            this.scene.add(line);

            material = new THREE.LineBasicMaterial({
                color: 0x00FF00,
            });

            points = [];
            points.push( new THREE.Vector3(0, 0, 0) );
            points.push( new THREE.Vector3(0, 1, 0) );

            geometry = new THREE.BufferGeometry().setFromPoints(points);
            line = new THREE.Line(geometry, material);
            this.scene.add(line);

            material = new THREE.LineBasicMaterial({
                color: 0x0000FF,
            });

            points = [];
            points.push( new THREE.Vector3(0, 0, 0) );
            points.push( new THREE.Vector3(0, 0, 1) );

            geometry = new THREE.BufferGeometry().setFromPoints(points);
            line = new THREE.Line(geometry, material);
            this.scene.add(line);
        }

        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
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

        // Setup the effect composer
        this.composer = new EffectComposer(this.renderer);
        this.passes = [];

        const gl = this.renderer.getContext();
        const samples = gl.getParameter(gl.SAMPLES);
        this.composer.renderTarget1.samples = samples;
        this.composer.renderTarget2.samples = samples;

        this.composer.renderTarget1.depthTexture = new THREE.DepthTexture(this.composer._width, this.composer._height);
        this.composer.renderTarget2.depthTexture = new THREE.DepthTexture(this.composer._width, this.composer._height);

        let renderPass = new LayerRenderPass(this.scene, this.camera, Layers.Base);
        renderPass.clear = false;
        renderPass.clearDepth = false;
        this.composer.addPass(renderPass);
        this.passes[Passes.BaseRenderPass] = renderPass;

        renderPass = new LayerRenderPass(this.scene, this.camera, Layers.NoShadows);
        renderPass.enabled = false;
        this.composer.addPass(renderPass);
        this.passes[Passes.NoShadowsRenderPass] = renderPass;

        renderPass = new LayerRenderPass(this.scene, this.camera, Layers.Top);
        renderPass.enabled = false;
        renderPass.clearDepth = true;
        this.composer.addPass(renderPass);
        this.passes[Passes.TopRenderPass] = renderPass;

        let outputPass = new OutputPass();
        this.composer.addPass(outputPass);
        this.passes[Passes.ColorConversionPass] = outputPass;

        this.toolbar = new Toolbar(this.domElement);

        // Scene controls
        const renderer = this.labelRenderer;

        this.cameraControl = new OrbitControls(this.camera, renderer.domElement);
        this.cameraControl.damping = 0.2;
        this.cameraControl.maxPolarAngle = Math.PI / 2.0 + 0.2;
        this.cameraControl.target = new THREE.Vector3(0, 0.5, 0);
        this.cameraControl.update();

        // Robot controls
        this.transformControls = new TransformControlsManager(this.toolbar, renderer.domElement, this.camera, this.scene);
        this.transformControls.addEventListener("dragging-changed", evt => this.cameraControl.enabled = !evt.value);

        // Events handling
        new ResizeObserver(() => this._onDomElementResized()).observe(this.domElement);
        renderer.domElement.addEventListener('mousedown', evt => this._onMouseDown(evt));
        renderer.domElement.addEventListener('mouseup', evt => this._onMouseUp(evt));
        renderer.domElement.addEventListener('mousemove', evt => this._onMouseMove(evt));
        renderer.domElement.addEventListener('wheel', evt => this._onWheel(evt));

        document.addEventListener("visibilitychange", () => {
            if (document.hidden) {
                this.clock.stop();
                this.clock.autoStart = true;
                this.renderingCallbackTime = null;
            }
        });

        this.activateLayer(0);
    }


    render() {
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

            this.renderingCallbackTime = null;
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


        if (this.renderingCallback != null)
        {
            if (this.renderingCallbackTime == null)
                this.renderingCallbackTime = this.clock.elapsedTime;

            if (this.renderingCallbackTimestep > 0.0)
            {
                const tmax = this.clock.elapsedTime;
                while (this.renderingCallbackTime < tmax)
                {
                    // Update the physics simulator
                    if (this.physicsSimulator != null) {
                        this.physicsSimulator.update(this.renderingCallbackTime);
                        this.physicsSimulator.synchronize();
                    }

                    this.renderingCallback(
                        this.renderingCallbackTimestep,
                        this.renderingCallbackTime + this.renderingCallbackTimestep
                    );

                    this.renderingCallbackTime += this.renderingCallbackTimestep;

                    if (this.renderingCallback == null)
                        break;
                }
            }
            else
            {
                // Update the physics simulator
                if (this.physicsSimulator != null) {
                    this.physicsSimulator.update(this.clock.elapsedTime);
                    this.physicsSimulator.synchronize();
                }

                this.renderingCallback(delta, this.clock.elapsedTime);
            }
        }
        else
        {
            // Update the physics simulator
            if (this.physicsSimulator != null) {
                this.physicsSimulator.update(this.clock.elapsedTime);
                this.physicsSimulator.synchronize();
            }
        }

        // Ensure that the camera isn't below the floor
        this.cameraControl.target.z = Math.max(this.cameraControl.target.z, 0.0);
        if (this.camera.position.z < 0.1) {
            this.camera.position.z = 0.1;
            this.cameraControl.update();
        }

        // Synchronize the robots (if necessary)
        const cameraPosition = new THREE.Vector3();
        this.camera.getWorldPosition(cameraPosition);

        for (const name in this.robots) {
            this.robots[name].synchronize(
                cameraPosition,
                this.domElement.clientWidth,
                this.interactionState != InteractionStates.Manipulation
            );
        }

        // Update the gaussians (if necessary)
        const viewMatrix = new THREE.Matrix3().setFromMatrix4(this.camera.matrixWorldInverse);
        for (const name in this.gaussians.objects) {
            this.gaussians.get(name)._update(viewMatrix);
        }


        // Physics simulator-related rendering
        this.renderer.setRenderTarget(this.composer.readBuffer);

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

        this.renderer.setRenderTarget(null);

        // Rendering
        this.composer.render(delta * 0.001);

        // Display the labels
        this.camera.layers.enable(31);
        this.labelRenderer.render(this.scene, this.camera);
        this.camera.layers.disable(31);

        // Update the logmap visualisation (if necessary)
        if (this.logmap) {
            const cameraOrientation = new THREE.Quaternion();
            this.camera.getWorldQuaternion(cameraOrientation);

            this.renderer.clearDepth();
            this.logmap.render(this.renderer, cameraOrientation);
        }

        // Request another animation frame
        if (!this.parameters.get('external_loop'))
        {
            if (!this.mustStop)
                requestAnimationFrame(() => this.render());
            else
                this.mustStop = false;
        }
    }


    _onDomElementResized() {
        const width = this.domElement.clientWidth;
        const height = this.domElement.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
        this.composer.setSize(width, height);

        if (this.labelRenderer != null)
            this.labelRenderer.setSize(width, height);
    }


    _onMouseDown(event) {
        if ((event.target.toolButtonFor != undefined) && (event.button == 0)) {
            const robot = event.target.toolButtonFor;
            robot.toggleGripper(event.target.toolIndex);
            event.preventDefault();
            return;
        }

        if ((event.button != 0) || this.transformControls.isDragging() || !this.controlsEnabled)
            return;

        this.didClick = true;

        const pointer = this._getPointerPosition(event);

        this.raycaster.setFromCamera(pointer, this.camera);

        let intersects = this.raycaster.intersectObjects(this.targets.meshes, false);

        if (intersects.length == 0) {
            const gaussianCenters = Object.keys(this.gaussians.objects).map(name => { return this.gaussians.get(name); });
            intersects = this.raycaster.intersectObjects(gaussianCenters, false);
        }

        if (intersects.length == 0) {
            const tcpTargets = [];
            for (let name in this.robots) {
                const robot = this.robots[name];
                for (let tool of robot.tools) {
                    if (tool.tcpTarget != null)
                        tcpTargets.push(tool.tcpTarget);
                }
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
                let object = intersection.object;

                let scalingEnabled = false;
                let listener = null;

                if (object.tag == 'target-mesh') {
                    if (!this.objectsManipulationEnabled || ((object.robot != null) && !object.robot.controlsEnabled))
                        return;

                    object = object.parent;
                    listener = object.listener;
                }
                else if (object.tag == 'gaussian-center') {
                    if (!this.objectsManipulationEnabled)
                        return;

                    object = object.gaussian;
                    scalingEnabled = true;
                    listener = object.listener;
                }
                else if (object.tag == 'tcp-target') {
                    if (!this.endEffectorManipulationEnabled)
                        return;
                }

                this.transformControls.attach(object, scalingEnabled, listener);

                if (object.robot != null)
                    this.kinematicChain = object.robot.getKinematicChainForTool(object.tool);

                this._switchToInteractionState(InteractionStates.Manipulation, { robot: object.robot });
            } else {
                intersection = null;
            }
        }

        if (intersection == null) {
            if (hoveredIntersection != null) {
                if (this.jointsManipulationEnabled) {
                    if (this.linksManipulationEnabled) {
                        this._switchToInteractionState(InteractionStates.LinkDisplacement);

                        const segment = this.hoveredRobot._getSegmentOfJoint(this.hoveredJoint);
                        const jointIndex = segment.joints.indexOf(this.hoveredJoint);
                        const joint = segment.visual.joints[jointIndex];

                        const direction = new THREE.Vector3();
                        this.camera.getWorldDirection(direction);

                        this.kinematicChain = this.hoveredRobot.getKinematicChainForJoint(this.hoveredJoint);

                        this.planarIkControls.setup(
                            this.hoveredRobot,
                            joint.worldToLocal(hoveredIntersection.point.clone()),
                            this.kinematicChain,
                            hoveredIntersection.point,
                            direction
                        );
                    } else if (this.forceImpulsesEnabled) {
                        const sim = this.physicsSimulator.simulation;
                        const bodyId = this.hoveredGroup.bodyId;

                        let force = hoveredIntersection.point.clone();
                        force.sub(this.camera.position);
                        force.multiplyScalar(this.forceImpulses);

                        sim.xfrc_applied[bodyId * 6] = force.x;
                        sim.xfrc_applied[bodyId * 6 + 1] = force.y;
                        sim.xfrc_applied[bodyId * 6 + 2] = force.z;
                    } else {
                        this._switchToInteractionState(InteractionStates.JointDisplacement);
                    }
                }
            } else if (this.interactionState != InteractionStates.Manipulation) {
                this._switchToInteractionState(InteractionStates.Default);
            }
        }

        event.preventDefault();
    }


    _onMouseUp(event) {
        if ((event.button != 0) || this.transformControls.isDragging() || !this.controlsEnabled) {
            return;
        }

        if ((this.interactionState == InteractionStates.JointDisplacement) ||
            (this.interactionState == InteractionStates.LinkDisplacement)) {
            const pointer = this._getPointerPosition(event);
            const [hoveredRobot, hoveredGroup] = this._getHoveredRobotGroup(pointer);

            this.kinematicChain = null;

            if (hoveredGroup != null) {
                this._switchToInteractionState(InteractionStates.JointHovering, { robot: hoveredRobot, group: hoveredGroup });
            } else {
                this._switchToInteractionState(InteractionStates.Default);
            }

            this.didClick = false;

            this.planarIkControls.u = null;

            return;

        } else if (this.interactionState == InteractionStates.Manipulation) {
            if (this.transformControls.wasUsed() || !this.didClick)
                return;

            this.transformControls.detach();
            this.kinematicChain = null;

            let hoveredRobot = null;
            let hoveredGroup = null;
            if (this.jointsManipulationEnabled) {
                const pointer = this._getPointerPosition(event);
                [hoveredRobot, hoveredGroup] = this._getHoveredRobotGroup(pointer);
            }

            if ((hoveredGroup != null) && hoveredRobot.controlsEnabled)
                this._switchToInteractionState(InteractionStates.JointHovering, { robot: hoveredRobot, group: hoveredGroup });
            else
                this._switchToInteractionState(InteractionStates.Default);

        } else {
            this._switchToInteractionState(InteractionStates.Default);
        }

        event.preventDefault();

        this.didClick = false;
    }


    _onMouseMove(event) {
        this.didClick = false;

        if (!this.controlsEnabled)
            return;

        if (this.transformControls.isEnabled()) {
            if (this.transformControls.isDragging() && (this.planarIkControls != null) && (this.hoveredRobot != null)) {
                this.kinematicChain.ik(
                    this.hoveredRobot.getEndEffectorDesiredTransforms(this.kinematicChain.tool),
                    null,
                    null,
                    5
                );
            }
            return;
        }

        if (!this.jointsManipulationEnabled)
            return;

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
            this.planarIkControls.process(this.raycaster);

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
                const tint = new THREE.Color().setRGB(245 / 255, 175 / 255, 154 / 255, THREE.SRGBColorSpace);

                object.originalMaterial = object.material;
                object.material = object.material.clone();
                object.material.color.r *= tint.r;
                object.material.color.g *= tint.g;
                object.material.color.b *= tint.b;
            }
        }

        const segment = robot._getSegmentOfJoint(group.jointId);
        if (segment != null) {
            this.hoveredRobot = robot;
            this.hoveredGroup = group;
            this.hoveredGroup.children.forEach(_highlight);

            this.hoveredJoint = this.hoveredGroup.jointId;
        }
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

        this.hoveredRobot = null;
        this.hoveredGroup = null;
        this.hoveredJoint = null;
    }


    _getHoveredRobotGroup(pointer) {
        this.raycaster.setFromCamera(pointer, this.camera);

        const meshes = Object.values(this.robots).filter((r) => r.controlsEnabled)
                                                 .map((r) => r.segments.map((segment) => segment.visual.meshes).flat()).flat()
                                                 .filter((mesh) => mesh.parent.jointId !== undefined);

        let intersects = this.raycaster.intersectObjects(meshes, false);
        if (intersects.length == 0)
            return [null, null];

        const intersection = intersects[0];

        const group = intersection.object.parent;
        const robot = Object.values(this.robots).filter((r) => r.links.indexOf(group.bodyId) >= 0)[0];

        return [robot, group];
    }


    _changeHoveredJointPosition(delta) {
        let ctrl = this.hoveredRobot.getControl();
        ctrl[this.hoveredRobot.joints.indexOf(this.hoveredJoint)] -= delta;
        this.hoveredRobot.setControl(ctrl);
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
                this.hoveredRobot = null;
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
                if (this.hoveredGroup != null) {
                    this._disableJointHovering();

                    if (this.controlEndedCallback != null)
                        this.controlEndedCallback();
                }
                break;
        }

        this.interactionState = interactionState;

        switch (this.interactionState) {
            case InteractionStates.Default:
                this.cameraControl.enableZoom = true;
                this.cameraControl.enabled = true;
                break;

            case InteractionStates.Manipulation:
                this.hoveredRobot = parameters.robot;
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

                if (this.controlStartedCallback != null)
                    this.controlStartedCallback();
                break;
        }
    }
}


function initViewer3D() {
    // Add some modules to the global scope, so they can be accessed by PyScript
    globalThis.three = THREE;
    globalThis.katex = katex;
    globalThis.Viewer3Djs = Viewer3D;
    globalThis.Shapes = Shapes;
    globalThis.Layers = Layers;
    globalThis.Passes = Passes;
    globalThis.OutlinePass = OutlinePass;
    globalThis.LayerRenderPass = LayerRenderPass;
    globalThis.RobotBuilder = RobotBuilder;
    globalThis.readFile = readFile;
    globalThis.writeFile = writeFile;

    globalThis.configs = {
        RobotConfiguration: RobotConfiguration,
        Panda: PandaConfiguration,
        PandaNoHand: PandaNoHandConfiguration,
        G1: G1Configuration,
        G1UpperBody: G1UpperBodyConfiguration,
        G1WithHands: G1WithHandsConfiguration,
        G1WithHandsUpperBody: G1WithHandsUpperBodyConfiguration,
    };

    globalThis.gaussians = {
        sigmaFromQuaternionAndScale: sigmaFromQuaternionAndScale,
        sigmaFromMatrix3: sigmaFromMatrix3,
        sigmaFromMatrix4: sigmaFromMatrix4,
        matrixFromSigma: matrixFromSigma,
    };
}


function initPyScript() {
    initViewer3D();

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
    script.src = 'https://pyscript.net/releases/2024.8.2/core.js';
    script.type = 'module';
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

export { G1Configuration, G1UpperBodyConfiguration, G1WithHandsConfiguration, G1WithHandsUpperBodyConfiguration, LayerRenderPass, Layers, OutlinePass, PandaConfiguration, PandaNoHandConfiguration, Passes, RobotBuilder, RobotConfiguration, Shapes, Viewer3D, downloadFiles, downloadG1Robot, downloadPandaRobot, downloadScene, getURL, initPyScript, initViewer3D, matrixFromSigma, readFile, sigmaFromMatrix3, sigmaFromMatrix4, sigmaFromQuaternionAndScale, writeFile };
