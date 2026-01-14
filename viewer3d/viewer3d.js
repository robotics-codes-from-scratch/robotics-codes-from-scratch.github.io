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
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
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
    if (!fileExists(filename))
        return null;

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


function readXmlFile(filename) {
    const content = readFile(filename, false);
    if (content == null)
        return null;

    const parser = new DOMParser();
    return parser.parseFromString(content, "text/xml");
}


function fileExists(filename) {
    try {
        const stat = mujoco.FS.stat(filename);
        return true;
    } catch (ex) {
        return false;
    }
}


function pathJoin() {
    let result = '';

    for (var i = 0; i < arguments.length; ++i) {
        let part = arguments[i];
        if (part.length == 0)
            continue;

        if (part[part.length - 1] == '/')
            part = part.substring(0, part.length - 1);

        if (i > 0)
            result += '/';

        result += part;
    }

    return result;
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


function getFirstElementByTag(parent, name) {
    const elements = parent.getElementsByTagName(name);
    if (elements.length > 0)
        return elements[0];

    return null;
}


function getElementByTagAndName(parent, tag, name) {
    const elements = parent.getElementsByTagName(tag);

    for (let i = 0; i < elements.length; ++i)
    {
        if (elements[i].getAttribute('name') == name)
            return elements[i];      
    }

    return null;
}


function getUniqueId(length=32) {
    let id = URL.createObjectURL(new Blob()).substr(-36);
    id = id.replaceAll('-', '');
    return id.substr(-length);
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



/**
 * Represents a kinematic chain (a sequence of joints/actuators) inside a robot.
 * Provides FK, IK and jacobian utilities restricted to that chain.
 */
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


    /**
     * Extract values corresponding to this kinematic chain from a full-robot vector.
     *
     * To perform the opposite operation, use 'project()'.
     *
     * @param {Array<number>} values - full robot joint/control values
     * @returns {Array<number>}
     */
    sample(values) {
        return this.joints.map(
            (v, i) => values[this.robot.joints.indexOf(v)]
        );
    }


    /**
     * Project chain-local values into a full-robot vector (others set to 0).
     *
     * To perform the opposite operation, use 'sample()'.
     *

    Parameters:
        values (array): The joint positions or control values of this
                        kinematic chain

    Returns:
        The values for the whole robot
    */
    project(values) {
        this.joints.map(
            (v, i) => this.robot.joints.indexOf(v)
        );

        const result = new Array(this.robot.joints.length);
        result.fill(0.0);

        for (let i = 0; i < this.joints.length; ++i)
        {
            const index = this.robot.joints.indexOf(this.joints[i]);
            result[index] = values[i];
        }

        return result;
    }


    /**
     * Compute forward kinematics for this chain.
     * @param {Array<number>|math.Matrix} positions - local joint positions
     * @param {THREE.Vector3|null} [offset=null] - optional offset from last joint
     * @returns {Array<number>} [px,py,pz,qx,qy,qz,qw]
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


    /**
     * Solve inverse kinematics to reach target pose mu.
     * @param {Array|math.Matrix} mu - target (3D position, 4D quat, or 7D transform)
     * @param {number|null} [nbJoints=null]
     * @param {THREE.Vector3|null} [offset=null]
     * @param {number} [limit=5] - max iterations
     * @param {number} [dt=0.01]
     * @param {number} [successDistance=1e-4]
     * @param {boolean} [damping=false]
     * @returns {boolean} true if converged
     */
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


    /**
     * Numerical Jacobian for this chain.
     * @param {Array|math.Matrix|number} positions
     * @param {THREE.Vector3|null} [offset=null]
     * @returns {math.Matrix}
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


    /**
     * Get joint positions for this chain.
     * @returns {Array<number>}
     */
    getJointPositions() {
        return this.robot._simulator.getJointPositions(this.joints);
    }


    /**
     * Get actuator controls for this chain.
     * @returns {Array<number>}
     */
    getControl() {
        return this.robot._simulator.getControl(this.actuators);
    }


    /**
     * Set local actuator controls (clamped), optionally updating joint positions when paused.
     * @param {Array<number>} control
     * @returns {void}
     */
    setControl(control) {
        const ctrl = control.map(
            (v, i) => (Math.abs(this.limits[i][0]) > 1e-6) || (Math.abs(this.limits[i][1]) > 1e-6) ?
                            Math.min(Math.max(v, this.limits[i][0]), this.limits[i][1]) :
                            v
        );

        const nbJoints = control.length;

        // If the simulator is paused, setting the control command doesn't have any effect
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



/**
 * Visual helper showing a joint's allowable motion (circular arc) and a label.
 * Extends THREE.Object3D and is added to the scene at construction time.
 */
class JointPositionHelper extends THREE.Object3D {

    /**
     * @param {THREE.Scene} scene - scene where the helper will be attached
     * @param {number} layer - rendering layer to use
     * @param {number} jointId - MuJoCo joint id
     * @param {number} jointIndex - index used in label (human-friendly)
     * @param {THREE.Vector3} axis - joint axis
     * @param {number} jointPosition - initial position angle
     * @param {boolean} [invert=false]
     * @param {number|string} [color=0xff0000]
     * @param {number} [offset=0.0]
     */
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
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 */



/**
 * Base class describing a tool attached to a robot (visuals, joints, tcp).
 */
class Tool {

    /**
     * Create an empty tool instance. Subclasses populate joints/visuals/etc.
     */
    constructor() {
        this.type = 'generic';

        this.joints = [];
        this.actuators = [];
        this.links = [];

        this.tcp = null;
        this.tcpTarget = null;

        this.names = {
            joints: [],
            actuators: [],
            links: [],
        };

        this.visual = {
            joints: [],
            links: [],
            meshes: [],
        };

        this.parent = -1;
    }

}


/**
 * Gripper tool with animated open/close sequences. Uses TWEEN for animations.
 * @extends Tool
 */
class Gripper extends Tool {

    /**
     * @param {Object} configuration - tool configuration describing states/closed joints
     */
    constructor(configuration) {
        super();

        this.type = 'gripper';

        this.state = null;
        this.holdingSomeObject = false;

        this.states = structuredClone(configuration.states);
        this.closedJoints = structuredClone(configuration.closedJoints);
        this.holdingThreshold = configuration.holdingThreshold;

        this.button = {
            object: null,
            element: null,
        };

        this.ctrl = [];
    }


    open() {
        this.state = 'opening';
        this.holdingSomeObject = false;

        const tool = this;

        const tweens = this._createTweens(this.states.opening);

        tweens[tweens.length - 1].onComplete(() => {
            tool.state = 'opened';
        });

        tweens[0].start();
    }


    close(simulator) {
        this.state = 'closing';
        this.holdingSomeObject = false;

        const tool = this;

        const tweens = this._createTweens(this.states.closing);

        tweens[tweens.length - 1].onComplete(() => {
            setTimeout(() => {
                    tool.state = 'closed';

                    const pos = simulator.getJointPositions(tool.joints);
                    let diff = 0.0;
                    for (let i = 0; i < tool.joints.length; ++i)
                        diff += Math.abs(pos[i] - tool.closedJoints[i]);

                    tool.holdingSomeObject = diff > tool.holdingThreshold;
                },
                100
            );
        });

        tweens[0].start();
    }


    isOpen() {
        return (this.state == 'opened');
    }


    isClosed() {
        return (this.state == 'closed');
    }


    isHoldingSomeObject() {
        return this.holdingSomeObject;
    }


    _createTweens(steps) {
        const tool = this;

        const start = {};
        for (let i = 0; i < this.actuators.length; ++i)
            start['actuator_' + i] = this.ctrl[i];

        const tweens = [];

        for (let j = 0; j < steps.length; ++j) {
            const step = steps[j];

            const end = {};
            for (let i = 0; i < this.actuators.length; ++i)
                end['actuator_' + i] = step[0][i];

            const tween = new TWEEN.Tween(start)
                .to(end, step[1])
                .onUpdate(object => {
                    for (let i = 0; i < tool.actuators.length; ++i)
                        tool.ctrl[i] = object['actuator_' + i];
                });

            if (j == steps.length - 1)
                tween.easing(TWEEN.Easing.Quadratic.Out);
            else
                tween.easing(TWEEN.Easing.Linear.None);

            if (j > 0)
                tweens[j - 1].chain(tween);

            tweens.push(tween);
        }

        return tweens;
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



const _tmpVector3$1 = new THREE.Vector3();
const _tmpQuaternion$1 = new THREE.Quaternion();
const _tmpQuaternion2 = new THREE.Quaternion();



/**
 * Base Robot wrapper that exposes high-level API for joint/control/IK operations.
 * Instances are created by the PhysicsSimulator when a MuJoCo scene is loaded.
 */
class Robot {

    /**
     * @param {string} name
     * @param {Object} configuration
     * @param {PhysicsSimulator} physicsSimulator
     */
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


    /**
     * Clean up robot visuals and toolbar resources. Call before removing robot.
     * @returns {void}
     */
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


    /**
     * Return an array of all THREE.Mesh objects composing the robot visuals and tools.
     * @returns {Array<THREE.Mesh>}
     */
    getMeshes() {
        return this.segments.map((segment) => segment.visual.meshes).flat().concat(
                    this.tools.map((tool) => tool.visual.meshes).flat()
        );
    }


    /**
     * Get current joint positions (array of numbers).
     * @returns {Array<number>}
     */
    getJointPositions() {
        return this._simulator.getJointPositions(this.joints);
    }


    /**
     * Set joint positions, clamped to joint limits, and update actuators.
     * @param {Array<number>} positions
     * @returns {void}
     */
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


    /**
     * Get current actuator control values.
     * @returns {Array<number>}
     */
    getControl() {
        return this._simulator.getControl(this.actuators);
    }


    /**
     * Set actuator control values (with limit clamping). If simulator is paused, sets joint positions instead.
     * @param {Array<number>} control
     * @returns {void}
     */
    setControl(control) {
        const limits = this.segments.map((segment) => segment.limits).flat();

        const ctrl = control.map(
            (v, i) => (Math.abs(limits[i][0]) > 1e-6) || (Math.abs(limits[i][1]) > 1e-6) ?
                            Math.min(Math.max(v, limits[i][0]), limits[i][1]) :
                            v
        );

        const nbJoints = control.length;

        // Don't use control commands too close to whatever is already used in the simulation,
        // as it can lead to instabilities
        const ref = this.getControl().slice(0, nbJoints);
        const diff = ref.map(
            (v, i) => Math.abs(v - ctrl[i])
        );

        if (math.norm(Array.from(diff)) < 1e-2)
            return;

        // If the simulator is paused, setting the control command doesn't have any effect
        if (this._simulator.paused)
            this._simulator.setJointPositions(ctrl, this.joints.slice(0, nbJoints));

        this._simulator.setControl(ctrl, this.actuators.slice(0, nbJoints));
    }


    /**
     * Get current joint velocities.
     * @returns {Array<number>}
     */
    getJointVelocities() {
        return this._simulator.getJointVelocities(this.joints);
    }


    /**
     * Return the center of mass of the robot's subtree rooted at robotRoot.
     * @returns {THREE.Vector3}
     */
    getCoM() {
        const bodyIdx = this.names.links.indexOf(this.configuration.robotRoot);
        return this._simulator.getSubtreeCoM(this.links[bodyIdx]);
    }


    /**
     * Return the robot's default joint pose as a Float32Array.
     * @returns {Float32Array}
     */
    getDefaultPose() {
        const pose = new Float32Array(this.joints.length);
        pose.fill(0.0);

        for (let name in this.configuration.defaultPose)
            pose[this.names.joints.indexOf(name)] = this.configuration.defaultPose[name];

        return pose;
    }


    /**
     * Apply the robot's default joint pose immediately.
     * @returns {void}
     */
    applyDefaultPose() {
        this.setJointPositions(this.getDefaultPose());
    }


    /**
     * Map actuators to their indices in this robot.
     * @param {Array<number>} actuators
     * @returns {Array<number>}
     */
    getActuatorIndices(actuators) {
        return actuators.map((actuator) => this.actuators.indexOf(actuator));
    }


    /**
     * Number of attached tools/end-effectors.
     * @returns {number}
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


    /* Returns the local position and orientation of a specific end-effector of the robot
    in an array of the form: [px, py, pz, qx, qy, qz, qw]
    */
    _getEndEffectorLocalTransforms(index=0) {
        const position = this.tools[index].tcp.position;
        const quaternion = this.tools[index].tcp.quaternion;

        return [
            position.x, position.y, position.z,
            quaternion.x, quaternion.y, quaternion.z, quaternion.w,
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


    _getToolControlPoint(index=0) {
        if (index < this.tools.length)
            return this.tools[index].tcp;

        return null;
    }


    /**
     * Create a KinematicChain instance for a joint index or name.
     * @param {number|string} joint
     * @returns {KinematicChain}
     */
    getKinematicChainForJoint(joint) {
        return new KinematicChain(this, joint);
    }


    /**
     * Create a KinematicChain for the tool at the given index.
     * @param {number} [index=0]
     * @returns {KinematicChain}
     */
    getKinematicChainForTool(index=0) {
        return new KinematicChain(this, null, index);
    }


    _isGripperOpen(index=0) {
        const tool = this.tools[index];
        if (tool.type != 'gripper')
            return false;

        return tool.isOpen();
    }


    _isGripperClosed(index=0) {
        const tool = this.tools[index];
        if (tool.type != 'gripper')
            return false;

        return tool.isClosed();
    }


    _isGripperHoldingSomeObject(index=0) {
        const tool = this.tools[index];
        if (tool.type != 'gripper')
            return false;

        return tool.isHoldingSomeObject();
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
                tool.state = this.configuration.tools[i].state;

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

                tool.ctrl = this._simulator.getControl(tool.actuators);
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

        if (stateName == 'closed')
            tool.close(this._simulator);
        else
            tool.open();
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
                        } else if (this.isGripperClosed(i)) { // || this.isGripperHoldingSomeObject(i)) {
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
        let tool = null;

        if (configuration.type == "gripper")
            tool = new Gripper(configuration);
        else
            tool = new Tool();

        tool.tcp = tcp;

        if (body != null)
            tool.links.push(body);

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


    /* Returns the local position and orientation of the end-effector of the robot
    in an array of the form: [px, py, pz, qx, qy, qz, qw]
    */
    getEndEffectorLocalTransforms() {
        return this._getEndEffectorLocalTransforms();
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


    getToolControlPoint() {
        return this._getToolControlPoint();
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


    /* Returns the local position and orientation of a specific end-effector of the robot
    in an array of the form: [px, py, pz, qx, qy, qz, qw]
    */
    getEndEffectorLocalTransforms(index=0) {
        return this._getEndEffectorLocalTransforms(index);
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


    getToolControlPoint(index=0) {
        return this._getToolControlPoint(index);
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
 * SPDX-FileCopyrightText: Copyright © 2024 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Allows to manipulate a specific body in the physics simulation.

The body must have a freejoint to be considered as manipulable.

A physical body is an Object3D, so you can manipulate it like one.
*/
class PhysicalBody extends THREE.Object3D {

    constructor(name, bodyId) {
        super();

        this.name = name;
        this.bodyId = bodyId;

        this._previousPosition = new THREE.Vector3();
        this._previousOrientation = new THREE.Quaternion();

        this._synchronize();
    }


    _apply(simulator) {
        if (!this.position.equals(this._previousPosition))
            simulator.setBodyPosition(this.bodyId, this.position);

        if (!this.quaternion.equals(this._previousOrientation))
            simulator.setBodyOrientation(this.bodyId, this.quaternion);
    }


    _synchronize() {
        this._previousPosition.copy(this.position);
        this._previousOrientation.copy(this.quaternion);
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




/**
 * Load a MuJoCo scene XML and create a PhysicsSimulator instance.
 *
 * @param {string} filename - path to the XML file inside MuJoCo FS
 * @param {Object|null} lightIntensities - optional mapping of light intensities
 * @returns {PhysicsSimulator|null}
 */
function loadScene(filename, lightIntensities) {
    // Retrieve some infos from the XML file (not exported by the MuJoCo API)
    const xmlDoc = readXmlFile(filename);
    if (xmlDoc == null)
        return null;

    const freeCameraSettings = getFreeCameraSettings(xmlDoc);
    const statistics = getStatistics(xmlDoc);
    const fogSettings = getFogSettings(xmlDoc);
    const headlightSettings = getHeadlightSettings(xmlDoc, filename);

    // Load in the state from XML
    let model = new mujoco.Model(filename);

    return new PhysicsSimulator(
        model, freeCameraSettings, statistics, fogSettings, headlightSettings, lightIntensities
    );
}



/**
 * Wrapper around MuJoCo model/state/simulation exposing helpers to step the
 * physics, synchronize transforms into THREE objects and create Robot wrappers.
 */
class PhysicsSimulator {

    /**
     * PhysicsSimulator wraps a MuJoCo Model/State/Simulation and provides
     * helpers to step the physics and synchronize transforms to THREE objects.
     *
     * @param {Object} model - MuJoCo Model instance (from mujoco.Model)
     * @param {Object} freeCameraSettings - parsed free camera settings from XML
     * @param {Object|null} statistics - optional statistics overrides from XML
     * @param {Object} fogSettings - fog/haze settings parsed from XML
     * @param {Object} headlightSettings - headlight settings parsed from XML
     * @param {Object|null} lightIntensities - optional mapping of light intensities
     */
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
        this.paused = false;
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


    /**
     * Free underlying MuJoCo objects and resources.
     * This should be called when the simulator is no longer needed.
     * @returns {void}
     */
    destroy() {
        this.simulation.delete();
        this.state.delete();
        this.model.delete();
    }


    /**
     * Step the simulation forward to match the provided time and apply any
     * per-step physical updates (e.g. applied forces reset).
     *
     * @param {number} time - target time (seconds) to update the simulation to
     * @returns {void}
     */
    update(time) {
        for (let b = 1; b < this.model.nbody; ++b) {
            const body = this.bodies[b];
            if (body instanceof PhysicalBody)
                body._apply(this);
        }

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


    /**
     * Synchronize the THREE.js scene objects with the current MuJoCo state.
     * This updates body/world transforms, light targets and any physical body
     * proxy objects.
     *
     * @returns {void}
     */
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

            if (body instanceof PhysicalBody)
                body._synchronize();
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


    /**
     * Return the names of bodies.
     * If indices is provided, returns the names for the requested body indices.
     *
     * @param {Array<number>|null} indices - optional array of body indices
     * @returns {Array<string>} array of body names
     */
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


    /**
     * Return the names of joints.
     * If indices is provided, returns the names for the requested joint indices.
     *
     * @param {Array<number>|null} indices - optional array of joint indices
     * @returns {Array<string>} array of joint names
     */
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


    /**
     * Return the names of actuators.
     * If indices is provided, returns the names for the requested actuator indices.
     *
     * @param {Array<number>|null} indices - optional array of actuator indices
     * @returns {Array<string>} array of actuator names
     */
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


    /**
     * Return the [min, max] range for a joint.
     *
     * @param {number} jointId - joint index
     * @returns {Array<number>} two-element array [min, max]
     */
     jointRange(jointId) {
        return this.model.jnt_range.slice(jointId * 2, jointId * 2 + 2);
    }


    /**
     * Return the control range [min, max] for an actuator.
     *
     * @param {number} actuatorId - actuator index
     * @returns {Array<number>} two-element array [min, max]
     */
    actuatorRange(actuatorId) {
        return this.model.actuator_ctrlrange.slice(actuatorId * 2, actuatorId * 2 + 2);
    }


    /**
     * Return the actuator index associated with a joint, or -1 if none.
     *
     * @param {number} jointId - joint index
     * @returns {number} actuator index or -1
     */
    getJointActuator(jointId) {
        const index = this.model.actuator_trnid.indexOf(jointId);
        if (index != -1)
            return index  / 2;

        return -1;
    }


    /**
     * Get joint positions (qpos) for all joints or a subset.
     *
     * @param {Array<number>|null} indices - optional joint indices to query
     * @returns {Float64Array} positions array (length = nb joints or indices.length)
     */
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


    /**
     * Set joint positions (qpos) for all joints or for a subset.
     * When setting a subset, positions is expected to be ordered according to indices.
     *
     * @param {Float64Array|Array<number>} positions - positions to assign
     * @param {Array<number>|null} indices - optional joint indices to set
     * @returns {void}
     */
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


    /**
     * Get actuator control signals for all actuators or a subset.
     *
     * @param {Array<number>|null} indices - optional actuator indices to query
     * @returns {Float64Array} control values
     */
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


    /**
     * Set actuator control signals for all actuators or a subset.
     *
     * @param {Float64Array|Array<number>} ctrl - control values to set
     * @param {Array<number>|null} indices - optional actuator indices to set
     * @returns {void}
     */
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


    /**
     * Get joint velocities (qvel) for all joints or a subset.
     *
     * @param {Array<number>|null} indices - optional joint indices to query
     * @returns {Float64Array} velocities array
     */
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


    /**
     * Return a PhysicalBody instance by name, or null if not found.
     *
     * @param {string} name - body name
     * @returns {PhysicalBody|null}
     */
    getPhysicalBody(name) {
        for (let b = 1; b < this.model.nbody; ++b) {
            const body = this.bodies[b];

            if (!(body instanceof PhysicalBody))
                continue;

            if (body.name == name)
                return body;
        }

        return null;
    }


    /**
     * Return the body index for a given body name, or null if not found.
     *
     * @param {string} name - body name
     * @returns {number|null} body index or null
     */
    getBodyId(name) {
        for (let b = 0; b < this.model.nbody; ++b) {
            const bodyName = this.names[this.model.name_bodyadr[b]];

            if (bodyName == name)
                return b;
        }

        return null;
    }


    /**
     * Return the world position of a body as a THREE.Vector3.
     *
     * @param {number} bodyId - body index
     * @returns {THREE.Vector3} position vector
     */
    getBodyPosition(bodyId) {
        const pos = new THREE.Vector3();
        this._getPosition(this.simulation.xpos, bodyId, pos);
        return pos;
    }


    /**
     * Set the world position for a body. This writes into the simulator qpos
     * and zeros the corresponding velocity components.
     *
     * @param {number} bodyId - body index
     * @param {THREE.Vector3} position - new world position
     * @returns {void}
     */
    setBodyPosition(bodyId, position) {
        const jntadr = this.model.body_jntadr[bodyId];

        const posIndex = this.model.jnt_qposadr[jntadr];
        this.simulation.qpos[posIndex] = position.x;
        this.simulation.qpos[posIndex + 1] = position.y;
        this.simulation.qpos[posIndex + 2] = position.z;

        const velIndex = this.model.jnt_dofadr[jntadr];
        this.simulation.qvel[velIndex] = 0.0;
        this.simulation.qvel[velIndex + 1] = 0.0;
        this.simulation.qvel[velIndex + 2] = 0.0;
    }


    /**
     * Return the world orientation of a body as a THREE.Quaternion.
     *
     * @param {number} bodyId - body index
     * @returns {THREE.Quaternion} orientation quaternion
     */
    getBodyOrientation(bodyId) {
        const quat = new THREE.Quaternion();
        this._getQuaternion(this.simulation.xquat, bodyId, quat);
        return quat;
    }


    /**
     * Set the world orientation (quaternion) for a body in the simulator qpos
     * and zero the corresponding angular velocity components.
     *
     * @param {number} bodyId - body index
     * @param {THREE.Quaternion} orientation - quaternion (w,x,y,z)
     * @returns {void}
     */
    setBodyOrientation(bodyId, orientation) {
        const jntadr = this.model.body_jntadr[bodyId];

        const posIndex = this.model.jnt_qposadr[jntadr] + 3;
        this.simulation.qpos[posIndex] = orientation.w;
        this.simulation.qpos[posIndex + 1] = orientation.x;
        this.simulation.qpos[posIndex + 2] = orientation.y;
        this.simulation.qpos[posIndex + 3] = orientation.z;

        const velIndex = this.model.jnt_dofadr[jntadr] + 3;
        this.simulation.qvel[velIndex] = 0.0;
        this.simulation.qvel[velIndex + 1] = 0.0;
        this.simulation.qvel[velIndex + 2] = 0.0;
    }


    /**
     * Return a THREE.Object3D representing the named site, or null if not found.
     *
     * @param {string} name - site name
     * @returns {THREE.Object3D|null}
     */
    getSite(name) {
        for (let s = 0; s < this.model.nsite; ++s) {
            const siteName = this.names[this.model.name_siteadr[s]];

            if (siteName == name)
                return this.sites[s];
        }

        return null;
    }


    /**
     * Return the names of all lights defined in the MuJoCo model.
     *
     * @returns {Array<string>} array of light names
     */
    lightNames() {
        const names = [];

        for (let j = 0; j < this.model.nlight; ++j)
            names.push(this.names[this.model.name_lightadr[j]]);

        return names;
    }


    /**
     * Return the center of mass of the subtree rooted at the given body.
     *
     * @param {number} bodyId - body index
     * @returns {THREE.Vector3} subtree center of mass
     */
    getSubtreeCoM(bodyId) {
        const pos = new THREE.Vector3();
        this._getPosition(this.simulation.subtree_com, bodyId, pos);
        return pos;
    }


    /**
     * Create a high-level Robot wrapper (SimpleRobot or ComplexRobot) based on
     * the provided infos. The returned robot object exposes convenience
     * methods to access joint indices, actuators and tools.
     *
     * @param {Object} infos - robot information object (name, configuration, ...)
     * @returns {SimpleRobot|ComplexRobot|null} robot instance or null on failure
     */
    createRobot(infos) {
        const sim = this;

        function _getChildBodies(bodyIdx, children, configuration) {
            for (let b = bodyIdx + 1; b < sim.model.nbody; ++b) {
                if (sim.names[sim.model.name_bodyadr[b]] == configuration.root)
                    continue;

                if (sim.model.body_parentid[b] == bodyIdx) {
                    children.push(b);
                    _getChildBodies(b, children, configuration);
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


        // Create the robot
        const robot = infos.configuration.tools.length > 1 ?
                        new ComplexRobot(infos.name, infos.configuration, this) :
                        new SimpleRobot(infos.name, infos.configuration, this);

        // Retrieve all the tools of the robot
        const toolBodies = [];
        for (let j = 0; j < infos.configuration.tools.length; ++j) {
            const cfg = infos.configuration.tools[j];

            let body = null;
            if (cfg.root != null)
            {
                body = _getBody(cfg.root);
                if (body == null) {
                    console.error("Failed to create the robot: link '" + cfg.root + "' not found");
                    return null;
                }

                toolBodies.push(body);
            }

            const tcp = _getSite(cfg.tcpSite);

            const tool = robot._createTool(tcp, body, cfg);

            if (body != null)
                _getChildBodies(body, tool.links, cfg);
        }

        // Retrieve all the segments of the robot
        const rootBody = _getBody(infos.configuration.robotRoot);
        if (rootBody == null) {
            console.error("Failed to create the robot: link '" + infos.configuration.robotRoot + "' not found");
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
            const cfg = infos.configuration.tools[j];

            let parentBody = null;
            if (cfg.root != null)
                parentBody = this.model.body_parentid[tool.links[0]];
            else
                parentBody = this.model.site_bodyid[tool.tcp.site_id];

            tool.parent = robot._getSegmentOfBody(parentBody);

            // Convert the ignored and inverted joint names to indices
            const jointNames = this.jointNames();

            if (cfg.ignoredJoints != undefined) {
                for (let k = 0; k < cfg.ignoredJoints.length; ++k)
                    tool.ignoredJoints.push(jointNames.indexOf(cfg.ignoredJoints[k]));
            }

            if (cfg.invertedJoints != undefined) {
                for (let k = 0; k < cfg.invertedJoints.length; ++k)
                    tool.invertedJoints.push(jointNames.indexOf(cfg.invertedJoints[k]));
            }

            tool.ctrl = this.getControl(tool.actuators);
        }

        // Let the robot initialise its internal state
        robot._init();

        return robot;
    }


    /**
     * Return the skybox/background texture(s) if present, otherwise null.
     * Skybox textures are returned as an array of 6 DataTexture objects.
     *
     * @returns {THREE.Texture|Array<THREE.Texture>|null}
     */
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
                let joint = null;
                for (let j = 0; j < this.model.njnt; ++j) {
                    if (this.model.jnt_bodyid[j] == b) {
                        if (this.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE.value)
                            joint = j;
                        break;
                    }
                }

                if (joint !== null) {
                    this.bodies[b] = new PhysicalBody(this.names[this.model.name_bodyadr[b]], b);
                } else {
                    this.bodies[b] = new THREE.Group();
                    this.bodies[b].name = this.names[this.model.name_bodyadr[b]];
                    this.bodies[b].bodyId = b;
                }

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
            let mesh = null;
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

        const names = this.lightNames();

        for (let l = 0; l < this.model.nlight; ++l) {
            let light = null;

            const intensity = lightIntensities[names[l]];

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


function getFreeCameraSettings(xmlDoc) {
    const settings = {
        fovy: 45.0,
        azimuth: 90.0,
        elevation: -45,
        znear: 0.01,
        zfar: 50,
    };

    const xmlVisual = getFirstElementByTag(xmlDoc, "visual");
    if (xmlVisual == null)
        return settings;

    const xmlGlobal = getFirstElementByTag(xmlVisual, "global");
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

    const xmlMap = getFirstElementByTag(xmlVisual, "map");
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

    const xmlStatistic = getFirstElementByTag(xmlDoc, "statistic");
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

    const xmlVisual = getFirstElementByTag(xmlDoc, "visual");
    if (xmlVisual == null)
        return settings;

    const xmlRgba = getFirstElementByTag(xmlVisual, "rgba");
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

    const xmlMap = getFirstElementByTag(xmlVisual, "map");
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

    const xmlVisual = getFirstElementByTag(xmlDoc, "visual");
    if (xmlVisual == null)
        return settings;

    let modified = false;

    const xmlHeadlight = getFirstElementByTag(xmlVisual, "headlight");
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

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 */



let Path$1 = class Path {
    constructor(path) {
        if (path instanceof Path)
            path = path.toString();

        if (typeof path !== "string") {
            throw new TypeError("The path must be a string");
        }

        this.separator = "/";
        this.isAbsolute = path.startsWith(this.separator);

        const rawParts = path.split(this.separator).filter(p => p.length > 0);
        this.parts = this._normalizeParts(rawParts, this.isAbsolute);
    }

    /**
     * Normalise the parts of a path, handling "." and ".."
     * @param {string[]} parts
     * @param {boolean} absolute
     * @returns {string[]}
     */
    _normalizeParts(parts, absolute) {
        const stack = [];

        for (const part of parts) {
            if (part === "." || part === "") {
                continue;
            } else if (part === "..") {
                if (stack.length > 0 && stack[stack.length - 1] !== "..") {
                    stack.pop();
                } else if (!absolute) {
                    stack.push("..");
                }
            } else {
                stack.push(part);
            }
        }

        return stack;
    }

    /**
     * Join a variable number of parts to the path and return the new path
     * @param  {...(string|Path)} parts
     * @returns {Path}
     */
    join(...parts) {
        let newParts = [...this.parts];
        let isAbsolute = this.isAbsolute;

        for (const part of parts) {
            let segmentParts;

            if (part instanceof Path) {
                segmentParts = part.parts;
            } else if (typeof part === "string") {
                segmentParts = part.split(this.separator).filter(p => p.length > 0);
            } else {
                throw new TypeError("Joined parts must be either strings or Paths");
            }

            newParts.push(...segmentParts);
        }

        const normalized = this._normalizeParts(newParts, isAbsolute);
        const prefix = isAbsolute ? this.separator : "";

        return new Path(prefix + normalized.join(this.separator));
    }

    /**
     * Returns the parent folder
     * @returns {Path}
     */
    dirname() {
        if (this.parts.length === 0) {
            return new Path(this.isAbsolute ? "/" : ".");
        }
        const dirParts = this.parts.slice(0, -1);
        const prefix = this.isAbsolute ? this.separator : "";
        return new Path(prefix + dirParts.join(this.separator));
    }

    /**
     * Returns the last segment (either a filename or a folder)
     * @returns {string}
     */
    basename() {
        if (this.parts.length === 0) return this.isAbsolute ? "/" : ".";
        return this.parts[this.parts.length - 1];
    }

    /**
     * Returns the extension of the file (including the point)
     * @returns {string}
     */
    extname() {
        const base = this.basename();
        const idx = base.lastIndexOf(".");
        if (idx <= 0) return "";
        return base.slice(idx);
    }

    isDirectory() {
        const base = this.basename();
        const idx = base.lastIndexOf(".");
        return (idx <= 0);
    }

    /**
     * Compute the relative path from another one
     * @param {string|Path} base
     * @returns {Path}
     */
    relativeTo(base) {
        if (typeof base === "string") base = new Path(base);
        if (!(base instanceof Path)) {
            throw new TypeError("The argument must be a Path or a string");
        }

        // Both paths must be either absolute or relatives
        if (this.isAbsolute !== base.isAbsolute) {
            throw new Error("Both paths must be either absolute or relatives");
        }

        // Find the first point of divergence
        let i = 0;
        while (i < this.parts.length && i < base.parts.length && this.parts[i] === base.parts[i]) {
            i++;
        }

        // Go up for each folder remaining in the base
        const up = base.parts.slice(i).map(() => "..");

        // Add the remaining parts from this path
        const down = this.parts.slice(i);

        const relativeParts = [...up, ...down];
        return new Path(relativeParts.join(this.separator) || ".");
    }

    /**
     * Returns the string representation of the path
     */
    toString() {
        const prefix = this.isAbsolute ? this.separator : "";
        return prefix + this.parts.join(this.separator);
    }

    mkdir() {
        if (!this.isAbsolute)
            throw new Error("The path must be absolute");

        if (!this.isDirectory()) {
            this.dirname().mkdir();
            return;
        }

        let current = this.separator + this.parts[0];

        let i = 0;
        while (i < this.parts.length) {
            if (!fileExists(current))
                mujoco.FS.mkdir(current);

            i++;
            if (i < this.parts.length)
                current += this.separator + this.parts[i];
        }
    }

    exists() {
        return fileExists(this.toString());
    }

    read(binary=false) {
        if (this.isDirectory())
            return null;

        return readFile(this.toString(), binary);
    }

    readXML() {
        if (this.isDirectory())
            return null;

        return readXmlFile(this.toString());
    }

    write(content) {
        if (this.isDirectory())
            return null;

        writeFile(this.toString(), content);
        return true;
    }
};

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



async function downloadFile(url, dstFolder) {
    const offset = url.lastIndexOf('/');
    const srcFolderUrl = url.substring(0, offset);
    const filename = url.substring(offset + 1);

    await downloadFiles(srcFolderUrl, dstFolder, [ filename ]);
}


/* Download some files and store them in Mujoco's filesystem

Parameters:
    dstFolder (string): The destination folder in Mujoco's filesystem
    srcFolderUrl (string): The URL of the folder in which all the files are located
    filenames ([string]): List of the filenames
*/
async function downloadFiles(srcFolderUrl, dstFolder, filenames) {
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

            if (!fileExists(path))
                FS.mkdir(path);
        }
    }

    if (srcFolderUrl[srcFolderUrl.length-1] != '/')
        srcFolderUrl += '/';

    for (let i = 0; i < filenames.length; ++i) {
        const filename = filenames[i];

        const srcFilenameUrl = pathJoin(srcFolderUrl, filename);
        const dstFilename = pathJoin(dstFolder, filename);

        if (fileExists(dstFilename))
            continue;

        const data = await fetch(srcFilenameUrl);
        if (data.ok) {
            const contentType = data.headers.get("content-type");
            if ((contentType == 'application/xml') || (contentType == 'text/plain')) {
                mujoco.FS.writeFile(dstFilename, await data.text());
            } else {
                mujoco.FS.writeFile(dstFilename, new Uint8Array(await data.arrayBuffer()));
            }
        } else {
            console.error("Failed to download the file: '" + srcFilenameUrl + "'");
        }
    }
}



/* Download a scene file

The scenes are stored in Mujoco's filesystem at '/scenes'
*/
async function downloadScene(url, destFolder='/scenes') {
    // Download the XML file of the scene
    const offset = url.lastIndexOf('/');
    const srcFolderUrl = url.substring(0, offset);
    const filename = url.substring(offset + 1);

    if (!(destFolder instanceof Path$1))
        destFolder = new Path$1(destFolder);

    const dstFilename = destFolder.join(filename);

    if (dstFilename.exists())
        return;

    await downloadFile(url, destFolder.toString());

    // Load the XML document
    let xmlDoc = dstFilename.readXML();
    if (xmlDoc == null)
    {
        console.error("Failed to load the file '" + dstFilename.toString() + "'");
        return;
    }

    // Download all the assets used by the scene
    const assets = getSceneAssets(xmlDoc);
    for (let folder in assets) {
        if (folder != '.')
            await downloadFiles(pathJoin(srcFolderUrl, folder), destFolder.join(folder).toString(), assets[folder]);
        else
            await downloadFiles(srcFolderUrl, destFolder.toString(), assets[folder]);
    }

    // Download all the included XML scenes
    const includedFiles = getIncludedFiles(xmlDoc);
    for (let includedFile of includedFiles) {
        includedFile = new Path$1(includedFile);
        if (!includedFile.isAbsolute)
            await downloadScene(pathJoin(srcFolderUrl, includedFile.toString()), destFolder.join(includedFile.dirname()));
    }
}



function getSceneAssets(xmlDoc) {
    const result = {};

    // Get assets folders
    let meshdir = null;
    let texturedir = null;

    let xmlCompiler = getFirstElementByTag(xmlDoc, 'compiler');
    if (xmlCompiler != null) {
        if (xmlCompiler.hasAttribute('meshdir'))
            meshdir = xmlCompiler.getAttribute('meshdir');

        if (xmlCompiler.hasAttribute('texturedir'))
            texturedir = xmlCompiler.getAttribute('texturedir');
    }

    // Get all meshes
    getAllAssetsOfType(xmlDoc, 'mesh', meshdir, result);

    // Get all textures
    getAllAssetsOfType(xmlDoc, 'texture', texturedir, result);

    return result;
}



function getAllAssetsOfType(xmlDoc, assetsType, assetsFolder, result) {
    const xmlAssets = xmlDoc.getElementsByTagName(assetsType);
    for (let xmlAsset of xmlAssets) {
        if (!xmlAsset.hasAttribute('file'))
            continue;

        let filename = xmlAsset.getAttribute('file');
        if (assetsFolder != null)
            filename = pathJoin(assetsFolder, filename);

        const offset = filename.lastIndexOf('/');
        let folder = '.';

        if (offset != -1) {
            folder = filename.substring(0, offset);
            filename = filename.substring(offset + 1);
        }

        if (result[folder] == undefined)
            result[folder] = [];

        result[folder].push(filename);
    }
}



function getIncludedFiles(xmlDoc) {
    const result = [];

    const xmlIncludes = xmlDoc.getElementsByTagName('include');
    for (let xmlInclude of xmlIncludes) {
        let filename = xmlInclude.getAttribute('file');
        result.push(filename);
    }

    return result;
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
/**
 * Manager/wrapper around three.js TransformControls that provides a small
 * toolbar to switch modes (translate/rotate/scale) and space (local/world).
 *
 * It also exposes a simple listener API for transform changes.
 *
 * @example
 * const manager = new TransformControlsManager(toolbar, renderer.domElement, camera, scene);
 * manager.attach(object, true, (obj, dragging) => {});
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


    /**
     * Forwards event listeners to the underlying TransformControls instance.
     *
     * @param {string} name - Event name (e.g. 'change', 'dragging-changed')
     * @param {Function} fct - Callback
     */
    addEventListener(name, fct) {
        this.transformControls.addEventListener(name, fct);
    }


    /**
     * Enable or disable the transform controls UI.
     * @param {boolean} enabled
     * @param {boolean} [withScaling=false] - show scaling option when true
     * @returns {void}
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


    /**
     * Returns whether transform controls are enabled.
     * @returns {boolean}
     */
    isEnabled() {
        return this.enabled;
    }


    /**
     * Returns whether a drag transform is currently active.
     * @returns {boolean}
     */
    isDragging() {
        return this.transformControls.dragging;
    }


    /**
     * Returns true if the controls were used since the last call, and resets the flag.
     * @returns {boolean}
     */
    wasUsed() {
        const result = this.used;
        this.used = false;
        return result;
    }


    /**
     * Attach a THREE.Object3D to the transform controls and show the UI.
     *
     * @param {THREE.Object3D} object - object to transform
     * @param {boolean} [withScaling=false] - show scaling option
     * @param {Function|null} [listener=null] - optional listener called (object, isActive)
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


    /**
     * Return the currently attached object (or null).
     * @returns {THREE.Object3D|null}
     */
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



/**
 * Small helper that provides planar IK support for dragging links.
 *
 * The class defines an infinite plane and when a raycaster intersects that
 * plane it computes an IK update using the provided KinematicChain.
 *
 * @example
 * const pik = new PlanarIKControls();
 * pik.setup(robot, offset, kinematicChain, startPosition, planeDirection);
 * pik.process(raycaster);
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


    /**
     * Configure the planar IK helper.
     * @param {Robot} robot
     * @param {THREE.Vector3} offset - local offset from link to target
     * @param {KinematicChain} kinematicChain - solver used to perform IK
     * @param {THREE.Vector3} startPosition - initial point on the plane
     * @param {THREE.Vector3} planeDirection - plane normal direction
     * @returns {void}
     */
    setup(robot, offset, kinematicChain, startPosition, planeDirection) {
        this.robot = robot;
        this.offset = offset;
        this.kinematicChain = kinematicChain;

        this.plane.position.copy(startPosition); 

        this.plane.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), planeDirection);
        this.plane.updateMatrixWorld(true);
    }


    /**
     * Process a raycaster intersection with the planar workspace and run IK if hit.
     * @param {THREE.Raycaster} raycaster
     * @returns {void}
     */
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

    /**
     * Reset internal state (no-op currently).
     * @returns {void}
     */
    reset() {
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
        } else if (dist < -1) {
            dist = -1;
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


/**
 * Available target shapes.
 * @readonly
 * @enum {symbol}
 */
const Shapes = Object.freeze({
    Cube: Symbol("cube"),
    Cone: Symbol("cone"),
    Sphere: Symbol("sphere"),
    Mesh: Symbol("mesh"),
});



/**
 * A manipulable target object used to define desired poses for tools/end-effectors.
 * Extends THREE.Object3D and is intended to be attached to TransformControls.
 *
 * @extends THREE.Object3D
 */
class Target extends THREE.Object3D {
    /**
     * Create a new Target.
     * @param {string} name
     * @param {THREE.Vector3} position
     * @param {THREE.Quaternion} orientation
     * @param {number|string} [color=0x0000aa]
     * @param {Shapes} [shape=Shapes.Cube]
     * @param {(target:Object)=>void|null} [listener=null]
     * @param {Map|Object|null} [parameters=null]
     * @param {Object|null} [targetslist=null] - optional target list to register meshes
     */
    constructor(
        name, position, orientation, color=0x0000aa, shape=Shapes.Cube, listener=null, parameters=null,
        targetslist=null
    ) {
        super();

        this.name = name;
        this.listener = listener;

        if (parameters == null)
            parameters = new Map();
        else if (!(parameters instanceof Map))
            parameters = new Map(Object.entries(parameters));

        if (shape == Shapes.Mesh) {
            const target = this;

            const url = parameters.get('url');
            const extension = url.toLowerCase().split('.').reverse()[0];

            const offset = parameters.get('offset') || [0.0, 0.0, 0.0];
            const orientation = parameters.get('orientation') || [0.0, 0.0, 0.0, 1.0];
            if (extension == 'obj') {
                const loader = new OBJLoader();

                loader.load(
                    url,

                    // called when resource is loaded
                    function(object) {
                        target._setup(object.children[0].geometry, color, parameters, offset, orientation, false);
                        targetslist.meshes.push(target.mesh);
                    }
                );
            } else if (extension == 'stl') {
                const loader = new STLLoader();

                loader.load(
                    url,

                    // called when resource is loaded
                    function(geometry) {
                        target._setup(geometry, color, parameters, offset, orientation, false);
                        targetslist.meshes.push(target.mesh);
                    }
                );
            }

        } else {
            // Create the geometry
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

                case Shapes.Mesh:
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

            // Create the mesh
            this._setup(geometry, color, parameters, [0.0, 0.0, 0.0], [0.707, 0, 0, 0.707]);
        }

        // Set the target position and orientation
        this.position.copy(position);
        this.quaternion.copy(orientation.clone().normalize());

        this.tag = 'target';
    }


    /**
     * Return the world transforms of the target as an array [px,py,pz,qx,qy,qz,qw].
     * @returns {number[]}
     */
    transforms() {
        return [
            this.position.x, this.position.y, this.position.z,
            this.quaternion.x, this.quaternion.y, this.quaternion.z, this.quaternion.w,
        ];
    }


    /**
     * Free GPU resources used by this target (geometries/materials).
     * @returns {void}
     */
    dispose() {
        if (this.mesh) {
            this.mesh.geometry.dispose();
            this.mesh.material.dispose();
        }

        if (this.line) {
            this.line.geometry.dispose();
            this.line.material.dispose();
        }
    }


    _disableVisibility(materials) {
        if (this.mesh) {
            this.mesh.material.colorWrite = false;
            this.mesh.material.depthWrite = false;

            materials.push(this.mesh.material);
        }

        if (this.line) {
            this.line.material.colorWrite = false;
            this.line.material.depthWrite = false;

            materials.push(this.line.material);
        }
    }


    _setup(geometry, color, parameters, position, orientation, line=true) {
        this.mesh = new THREE.Mesh(
            geometry,
            new THREE.MeshBasicMaterial({
                color: color,
                opacity: parameters.get('opacity') || 0.75,
                transparent: true
            })
        );

        const q = new THREE.Quaternion(orientation[0], orientation[1], orientation[2], orientation[3]);
        this.mesh.setRotationFromQuaternion(q);

        this.mesh.translateX(position[0]);
        this.mesh.translateY(position[1]);
        this.mesh.translateZ(position[2]);

        this.mesh.castShadow = true;
        this.mesh.receiveShadow = false;
        this.mesh.layers = this.layers;

        this.add(this.mesh);

        // Add a wireframe on top of the mesh
        if (line) {
            const wireframe = new THREE.WireframeGeometry(geometry);

            this.line = new THREE.LineSegments(wireframe);
            this.line.material.depthTest = true;
            this.line.material.opacity = 0.5;
            this.line.material.transparent = true;
            this.line.layers = this.layers;

            this.mesh.add(this.line);
        }

        this.mesh.tag = 'target-mesh';
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
        const target = new Target(name, position, orientation, color, shape, listener, parameters, this);
        this.add(target);
        return target;
    }


    /* Add a target to the list.

    Parameters:
        target (Target): The target
    */
    add(target) {
        this.targets[target.name] = target;

        if (target.mesh)
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



let CYLINDER_GEOMETRY = null;
let CONE_GEOMETRY = null;

const axis = new THREE.Vector3();


/* Visual representation of an arrow.

An arrow is an Object3D, so you can manipulate it like one.
*/
/**
 * Visual arrow helper composed of a cylinder + cone. Extends THREE.Object3D.
 */
class Arrow extends THREE.Object3D {

    /**
     * @param {string} name
     * @param {THREE.Vector3} origin
     * @param {THREE.Vector3} direction - unit vector
     * @param {number} [length=1]
     * @param {number|string} [color=0xffff00]
     * @param {boolean} [shading=false]
     * @param {number} [headLength=length*0.2]
     * @param {number} [headWidth=headLength*0.2]
     * @param {number} [radius=headWidth*0.3]
     */
    constructor(name, origin, direction, length=1, color=0xffff00, shading=false, headLength=length * 0.2,
        headWidth=headLength * 0.2, radius=headWidth*0.1
    ) {
        super();

        this.name = name;

        if (CYLINDER_GEOMETRY == null) {
            CYLINDER_GEOMETRY = new THREE.CylinderGeometry(1, 1, 1, 16);
            CYLINDER_GEOMETRY.translate(0, 0.5, 0);

            CONE_GEOMETRY = new THREE.ConeGeometry(0.5, 1, 16);
            CONE_GEOMETRY.translate(0, -0.5, 0);
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

        this.cylinder = new THREE.Mesh(CYLINDER_GEOMETRY, material);
        this.cylinder.layers = this.layers;

        this.add(this.cylinder);

        this.cone = new THREE.Mesh(CONE_GEOMETRY, material);
        this.cone.layers = this.layers;

        this.add(this.cone);

        this.position.copy(origin);
        this.setDirection(direction);
        this.setDimensions(length, headLength, headWidth, radius);
    }


    /* Sets the direction of the arrow
    */
    /**
     * Set the arrow direction. Expects a unit vector.
     * @param {THREE.Vector3} direction
     * @returns {void}
     */
    setDirection(direction) {
        // 'direction' is assumed to be normalized
        if (direction.y > 0.99999) {
            this.quaternion.set(0, 0, 0, 1);
        } else if (direction.y < -0.99999) {
            this.quaternion.set(1, 0, 0, 0);
        } else {
            axis.set(direction.z, 0, -direction.x).normalize();
            const radians = Math.acos(direction.y);
            this.quaternion.setFromAxisAngle(axis, radians);
        }
    }


    /**
     * Set the length and relative head size of the arrow.
     * @param {number} length
     * @param {number} [headLength=length*0.2]
     * @param {number} [headWidth=headLength*0.2]
     * @param {number} [radius=headWidth*0.3]
     * @returns {void}
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


    /**
     * Set the arrow color.
     * @param {number|string} color
     * @returns {void}
     */
    setColor(color) {
        this.line.material.color.set(color);
        this.cone.material.color.set(color);
    }


    /**
     * Dispose GPU resources used by this arrow.
     * @returns {void}
     */
    dispose() {
        if (this.cylinder.geometry != CYLINDER_GEOMETRY)
            this.cylinder.geometry.dispose();

        if (this.cone.geometry != CONE_GEOMETRY)
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
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



const AXES_MATERIALS = [];
const AXES_GEOMETRIES = [];


/* Visual representation of 3 XYZ axes.

Axes are an Object3D, so you can manipulate them like one.
*/
/**
 * Simple 3-axis visual helper (X:red, Y:green, Z:blue).
 * Extends THREE.Object3D.
 */
class Axes extends THREE.Object3D {

    /**
     * @param {string} name
     * @param {THREE.Vector3|null} [position=null]
     * @param {THREE.Quaternion|null} [orientation=null]
     * @param {number} [length=0.1]
     */
    constructor(name, position=null, orientation=null, length=0.1) {
        super();

        this.name = name;

        if (position != null)
            this.position.copy(position);

        if (orientation != null)
            this.quaternion.copy(orientation.clone().normalize());

        if (AXES_GEOMETRIES.length == 0)
        {
            AXES_MATERIALS.push(new THREE.LineBasicMaterial({
                color: 0xFF0000,
            }));

            AXES_MATERIALS.push(new THREE.LineBasicMaterial({
                color: 0x009900,
            }));

            AXES_MATERIALS.push(new THREE.LineBasicMaterial({
                color: 0x0000FF,
            }));

            let points = [];
            points.push( new THREE.Vector3(0, 0, 0) );
            points.push( new THREE.Vector3(1, 0, 0) );

            AXES_GEOMETRIES.push(new THREE.BufferGeometry().setFromPoints(points));

            points = [];
            points.push( new THREE.Vector3(0, 0, 0) );
            points.push( new THREE.Vector3(0, 1, 0) );
            AXES_GEOMETRIES.push(new THREE.BufferGeometry().setFromPoints(points));

            points = [];
            points.push( new THREE.Vector3(0, 0, 0) );
            points.push( new THREE.Vector3(0, 0, 1) );
            AXES_GEOMETRIES.push(new THREE.BufferGeometry().setFromPoints(points));
        }

        this.lines = [];

        for (let i = 0; i < 3; ++i)
        {
            let line = new THREE.Line(AXES_GEOMETRIES[i], AXES_MATERIALS[i]);
            line.scale.set(length, length, length);
            this.add(line);
            this.lines.push(line);
        }
    }


    /* Frees the GPU-related resources allocated by this instance. Call this method whenever this
    instance is no longer used in your app.
    */
    dispose() {
        for (let i = 0; i < 3; ++i)
        {
            if (this.lines[i].geometry != AXES_GEOMETRIES[i])
                this.lines[i].geometry.dispose();

            if (this.lines[i].material != AXES_MATERIALS[i])
                this.lines[i].material.dispose();
        }
    }


    _disableVisibility(materials) {
        for (let i = 0; i < 3; ++i)
        {
            if (this.lines[i].material == AXES_MATERIALS[i])
                this.lines[i].material = this.lines[i].material.clone();

            this.lines[i].material.colorWrite = false;
            this.lines[i].material.depthWrite = false;

            materials.push(this.lines[i].material);
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



/* Visual representation of a path.

A path is an Object3D, so you can manipulate it like one.
*/
/**
 * Visual representation of a path (tube) defined by an ordered list of points.
 * Extends THREE.Object3D.
 */
class Path extends THREE.Object3D {

    /**
     * @param {string} name
     * @param {Array<THREE.Vector3|number[]>} points
     * @param {number} [radius=0.01]
     * @param {number|string} [color=0xffff00]
     * @param {boolean} [shading=false]
     * @param {boolean} [transparent=false]
     * @param {number} [opacity=0.5]
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


    /**
     * Dispose GPU resources used by the path.
     * @returns {void}
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
/**
 * Visual representation of a single point (sphere) with optional KaTeX label.
 * Extends THREE.Object3D.
 */
class Point extends THREE.Object3D {

    /**
     * Create a point marker.
     * @param {string} name
     * @param {THREE.Vector3} position
     * @param {number} [radius=0.01]
     * @param {number|string} [color=0xffff00]
     * @param {string|null} [label=null] - LaTeX string rendered with KaTeX
     * @param {boolean} [shading=false]
     * @param {boolean} [transparent=false]
     * @param {number} [opacity=0.5]
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


    /**
     * Set the texture of the point's material from a URL.
     * @param {string} url
     * @returns {void}
     */
    setTexture(url) {
        const texture = new THREE.TextureLoader().load(url);
        this.mesh.material.map = texture;
    }


    /**
     * Dispose GPU resources used by this point.
     * @returns {void}
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



let SPHERE_GEOMETRY = null;


/* Computes the covariance matrix of a gaussian from an orientation and scale

Parameters:
    quaternion (Quaternion): The orientation
    scale (Vector3): The scale
*/
/**
 * Compute covariance matrix sigma from an orientation quaternion and a scale vector.
 * @param {THREE.Quaternion} quaternion
 * @param {THREE.Vector3} scale
 * @returns {THREE.Matrix3}
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
/**
 * Compute covariance sigma = RG * RG^T from a 3x3 rotation-scale matrix.
 * @param {THREE.Matrix3} matrix
 * @returns {THREE.Matrix3}
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
/**
 * Compute covariance from the upper-left 3x3 part of a Matrix4.
 * @param {THREE.Matrix4} matrix
 * @returns {THREE.Matrix3}
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
/**
 * Given a covariance matrix (Matrix3) compute a rotation-scale matrix RG such that
 * sigma = RG * RG^T. Uses eigen-decomposition via mathjs.
 *
 * @param {THREE.Matrix3} sigma
 * @returns {THREE.Matrix3}
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
/**
 * Visual representation of a 3D Gaussian. The object's scale and rotation encode
 * the covariance matrix, and the position encodes the mean (mu).
 *
 * @extends THREE.Object3D
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

        if (SPHERE_GEOMETRY == null) {
            SPHERE_GEOMETRY = new THREE.SphereGeometry(1.0, 32, 16);
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

        this.sphere = new THREE.Mesh(SPHERE_GEOMETRY, material);
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
        if (this.sphere.geometry != SPHERE_GEOMETRY)
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
/**
 * Haze/top-of-skybox helper implemented as a truncated cone mesh with vertex colors.
 * Extends THREE.Object3D.
 */
class Haze extends THREE.Object3D {

    /**
     * @param {number} nbSlices
     * @param {number} proportion
     * @param {{r:number,g:number,b:number}} color
     */
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

        if (robot._isGripperClosed())
            this.btn.innerText = 'Open gripper';
        else
            this.btn.innerText = 'Close gripper';

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
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
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
        this.filename = null;

        // Root link of the robot
        this.robotRoot = null;

        // Default pose of the robot
        this.defaultPose = {
        },

        this.jointPositionHelpers = {
            // Offsets to apply to the joint position helpers
            offsets: {},

            // Joint position helpers that must be inverted
            inverted: [],
        };

        this.defaultToolConfigurations = null;
    }


    addPrefix(prefix) {
        const configuration = new RobotConfiguration();

        configuration.filename = this.filename;

        configuration.robotRoot = prefix + this.robotRoot;

        for (let name in this.defaultPose)
            configuration.defaultPose[prefix + name] = this.defaultPose[name];

        for (let name in this.jointPositionHelpers.offsets)
            configuration.jointPositionHelpers.offsets[prefix + name] = this.jointPositionHelpers.offsets[name];

        for (let name of this.jointPositionHelpers.inverted)
            configuration.jointPositionHelpers.inverted.push(prefix + name);

        if (this.defaultToolConfigurations != null) {
            configuration.defaultToolConfigurations = [];

            for (let i = 0; i < this.defaultToolConfigurations.length; ++i)
                configuration.defaultToolConfigurations.push(this.defaultToolConfigurations[i].addPrefix(prefix));
        }

        return configuration;
    }
}


class ToolConfiguration {
    constructor() {
        this.filename = null;

        // Root link of the tool of the robot (can be null)
        this.root = null;

        // Site to use as the tool control point
        this.tcpSite = null;

        // Size of the TCP collision object
        this.tcpSize = 0.1;

        // Type of the tool ("generic", "gripper")
        this.type = "generic";
    }


    addPrefix(prefix, destination=null) {
        if (destination == null)
            destination = new ToolConfiguration();

        destination.filename = this.filename;
        destination.root = (this.root != null ? prefix + this.root : null);
        destination.tcpSite = (this.tcpSite != null ? prefix + this.tcpSite : null);
        destination.tcpSize = this.tcpSize;

        return destination;
    }
}


class GripperConfiguration extends ToolConfiguration {
    constructor() {
        super();

        this.type = "gripper";

        // Location of the button, relative to the tcp
        this.buttonOffset = [0, 0, 0];

        this.state = "closed";

        this.states = {
            opened: [],
            closed: [],
            opening: [],
            closing: [],
        };

        this.closedJoints = [];
        this.holdingThreshold = 1e-2;
    }


    addPrefix(prefix, destination=null) {
        if (destination == null)
            destination = new GripperConfiguration();

        destination.buttonOffset = this.buttonOffset;
        destination.state = this.state;
        destination.states = this.states;
        destination.closedJoints = this.closedJoints;
        destination.holdingThreshold = this.holdingThreshold;

        return super.addPrefix(prefix, destination);
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

        this.filename = 'franka_emika/panda/panda.xml';

        this.robotRoot = "link0";

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

        this.defaultToolConfigurations = [
            new DefaultPandaToolConfiguration()
        ];
    }
}


class DefaultPandaToolConfiguration extends ToolConfiguration {
    constructor() {
        super();

        this.tcpSite = 'attachment_site';
    }
}


class FrankaHandConfiguration extends GripperConfiguration {
    constructor() {
        super();

        this.filename = 'franka_emika/hand/hand.xml';

        this.root = "hand";
        this.tcpSite = "tcp";
        this.tcpSize = 0.1;
        this.buttonOffset = [0, -0.11, 0.05];

        this.state = 'closed';

        this.states = {
            opened: [255],
            closed: [0],
            opening: [
                [[255], 400],
            ],
            closing: [
                [[0], 400],
            ],
        };

        this.closedJoints = [0.0, 0.0];
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 */



class InspireRightRH56DFXConfiguration extends GripperConfiguration {
    constructor() {
        super();

        this.filename = 'inspire_robots/RH56DFX/right.xml';

        this.root = 'hand';
        this.tcpSite = 'tcp';
        this.tcpSize = 0.08;

        this.buttonOffset = [0.07, 0.0, -0.1];

        this.state = "opened";

        this.states = {
            opened: [0.0, 0.0, 0.0, 0.0, 0.0, -0.0454, 0.0, -0.0454, 0.0 -0.0454, 0.0, -0.0454],
            closed: [0.0, 0.0, 0.0, 0.0, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56],
            opening: [
                [[0.0, 0.0, 0.0, 0.0, 1.47, -0.0454, 1.47, -0.0454, 1.47, -0.0454, 1.47, -0.0454], 200],
                [[0.0, 0.0, 0.0, 0.0, 0.0, -0.0454, 0.0, -0.0454, 0.0 -0.0454, 0.0, -0.0454], 200],
            ],
            closing: [
                [[0.0, 0.0, 0.0, 0.0, 1.47, -0.0454, 1.47, -0.0454, 1.47, -0.0454, 1.47, -0.0454], 200],
                [[0.0, 0.0, 0.0, 0.0, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56], 200],
            ],
        };

        this.closedJoints = [0.0, 0.0, 0.0, 0.0, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56];
        this.holdingThreshold = 0.2;
    }
}


class InspireLeftRH56DFXConfiguration extends GripperConfiguration {
    constructor() {
        super();

        this.filename = 'inspire_robots/RH56DFX/left.xml';

        this.root = 'hand';
        this.tcpSite = 'tcp';
        this.tcpSize = 0.08;

        this.buttonOffset = [0.07, 0.0, -0.1];

        this.state = "opened";

        this.states = {
            opened: [0.0, 0.0, 0.0, 0.0, 0.0, -0.0454, 0.0, -0.0454, 0.0 -0.0454, 0.0, -0.0454],
            closed: [0.0, 0.0, 0.0, 0.0, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56],
            opening: [
                [[0.0, 0.0, 0.0, 0.0, 1.47, -0.0454, 1.47, -0.0454, 1.47, -0.0454, 1.47, -0.0454], 200],
                [[0.0, 0.0, 0.0, 0.0, 0.0, -0.0454, 0.0, -0.0454, 0.0 -0.0454, 0.0, -0.0454], 200],
            ],
            closing: [
                [[0.0, 0.0, 0.0, 0.0, 1.47, -0.0454, 1.47, -0.0454, 1.47, -0.0454, 1.47, -0.0454], 200],
                [[0.0, 0.0, 0.0, 0.0, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56], 200],
            ],
        };

        this.closedJoints = [0.0, 0.0, 0.0, 0.0, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56, 1.47, 1.56];
        this.holdingThreshold = 0.2;
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 */



class Robotiq2F85Configuration extends GripperConfiguration {
    constructor() {
        super();

        this.filename = 'robotiq/2f85/2f85.xml';

        this.root = "base";
        this.tcpSite = "tcp";
        this.tcpSize = 0.1;
        this.buttonOffset = [-0.11, 0, 0.05];

        this.state = "opened";

        this.states = {
            opened: [0],
            closed: [255],
            opening: [
                [[0], 400],
            ],
            closing: [
                [[255], 400],
            ],
        };

        this.closedJoints = [0.810, 0.810, 0.810, 0.810, 0.810, 0.810];
    }
}


class RobotiqHandEConfiguration extends GripperConfiguration {
    constructor() {
        super();

        this.filename = 'robotiq/hand-e/hande.xml';

        this.root = "root";
        this.tcpSite = "tcp";
        this.tcpSize = 0.1;
        this.buttonOffset = [-0.11, 0, 0.05];

        this.state = "opened";

        this.states = {
            opened: [0],
            closed: [255],
            opening: [
                [[0], 400],
            ],
            closing: [
                [[255], 400],
            ],
        };

        this.closedJoints = [0.0, 0.0];
    }
}

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 */



class UR5Configuration extends RobotConfiguration {
    constructor() {
        super();

        this.filename = 'universal_robots/ur5e/ur5e.xml';

        this.robotRoot = "base";

        this.defaultPose = {
            shoulder_pan_joint: -1.5708,
            shoulder_lift_joint: -1.5708,
            elbow_joint: 1.5708,
            wrist_1_joint: -1.5708,
            wrist_2_joint: -1.5708,
            wrist_3_joint: 0.0,
        };

        this.defaultToolConfigurations = [
            new DefaultUR5ToolConfiguration()
        ];
    }
}


class DefaultUR5ToolConfiguration extends ToolConfiguration {
    constructor() {
        super();

        this.tcpSite = 'attachment_site';
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

        this.filename = 'unitree/g1/g1.xml';

        this.robotRoot = "pelvis";

        this.defaultToolConfigurations = [
            new DefaultG1ToolConfiguration("right_tcp"),
            new DefaultG1ToolConfiguration("left_tcp"),
            new DefaultG1ToolConfiguration("right_foot_tcp"),
            new DefaultG1ToolConfiguration("left_foot_tcp"),
        ];
    }
}


class G1FixedLegsConfiguration extends RobotConfiguration {
    constructor() {
        super();

        this.filename = 'unitree/g1/g1_fixed_legs.xml';

        this.robotRoot = "pelvis";

        this.defaultToolConfigurations = [
            new DefaultG1ToolConfiguration("right_tcp"),
            new DefaultG1ToolConfiguration("left_tcp"),
        ];
    }
}


class DefaultG1ToolConfiguration extends ToolConfiguration {
    constructor(tcpSite) {
        super();

        this.tcpSite = tcpSite;
        this.tcpSize = 0.08;
    }
}


class UnitreeRightHandConfiguration extends ToolConfiguration {
    constructor() {
        super();

        this.filename = 'unitree/hands/right.xml';

        this.root = 'hand';
        this.tcpSite = 'tcp';
        this.tcpSize = 0.08;
    }
}


class UnitreeLeftHandConfiguration extends ToolConfiguration {
    constructor() {
        super();

        this.filename = 'unitree/hands/left.xml';

        this.root = 'hand';
        this.tcpSite = 'tcp';
        this.tcpSize = 0.08;
    }
}


class UnitreeRightDex31Configuration extends GripperConfiguration {
    constructor() {
        super();

        this.filename = 'unitree/dex3_1/right.xml';

        this.root = 'hand';
        this.tcpSite = 'tcp';
        this.tcpSize = 0.08;

        this.buttonOffset = [0.1, -0.07, 0];

        this.state = "opened";

        this.states = {
            opened: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            closed: [0.0, -1.05, -1.75, 1.57, 1.75, 1.57, 1.75],
            opening: [
                [[0.0, 0.0, -1, 1.57, 0.75, 1.57, 0.75], 200],
                [[0.0, 0.0, -0.75, 0.0, 0.75, 0.0, 0.75], 200],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 100],
            ],
            closing: [
                [[0.0, 0.0, -0.75, 0.0, 0.75, 0.0, 0.75], 100],
                [[0.0, 0.0, -1, 1.57, 0.75, 1.57, 0.75], 200],
                [[0.0, -1.05, -1.75, 1.57, 1.75, 1.57, 1.75], 200],
            ],
        };

        this.closedJoints = [-0.012, 0.0, -1.03, -1.34, 1.57, 1.72, 1.57, 1.72];
        this.holdingThreshold = 0.2;
    }
}


class UnitreeLeftDex31Configuration extends GripperConfiguration {
    constructor() {
        super();

        this.filename = 'unitree/dex3_1/left.xml';

        this.root = 'hand';
        this.tcpSite = 'tcp';
        this.tcpSize = 0.08;

        this.buttonOffset = [0.1, 0.07, 0];

        this.state = "opened";

        this.states = {
            opened: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            closed: [0.0, 1.05, 1.75, -1.57, -1.75, -1.57, -1.75],
            opening: [
                [[0.0, 0.0, 1.0, -1.57, -0.75, -1.57, -0.75], 200],
                [[0.0, 0.0, 0.75, -0, -0.75, -0, -0.75], 200],
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 100],
            ],
            closing: [
                [[0.0, 0.0, 0.75, 0.0, -0.75, 0.0, -0.75], 100],
                [[0.0, 0.0, 1.0, -1.57, -0.75, -1.57, -0.75], 200],
                [[0.0, 1.05, 1.75, -1.57, -1.75, -1.57, -1.75], 200],
            ],
        };

        this.closedJoints = [0.012, 0.0, 1.03, 1.34, -1.57, -1.72, -1.57, -1.72];
        this.holdingThreshold = 0.2;
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



const Robots = Object.freeze({
    Panda: Symbol("Panda"),
    G1: Symbol("G1"),
    G1_FixedLegs: Symbol("G1_FixedLegs"),
    UR5: Symbol("UR5"),
});


const Tools = Object.freeze({
    FrankaHand: Symbol("FrankaHand"),
    InspireRightRH56DFX: Symbol("InspireRightRH56DFX"),
    InspireLeftRH56DFX: Symbol("InspireLeftRH56DFX"),
    Robotiq_2F85: Symbol("Robotiq_2F85"),
    RobotiqHandE: Symbol("RobotiqHandE"),
    UnitreeRightHand: Symbol("UnitreeRightHand"),
    UnitreeLeftHand: Symbol("UnitreeLeftHand"),
    UnitreeRightDex3_1: Symbol("UnitreeRightDex3_1"),
    UnitreeLeftDex3_1: Symbol("UnitreeLeftDex3_1"),
});


const Configurations = {};

Configurations[Robots.Panda] = PandaConfiguration;
Configurations[Robots.G1] = G1Configuration;
Configurations[Robots.G1_FixedLegs] = G1FixedLegsConfiguration;
Configurations[Robots.UR5] = UR5Configuration;

Configurations[Tools.FrankaHand] = FrankaHandConfiguration;
Configurations[Tools.InspireRightRH56DFX] = InspireRightRH56DFXConfiguration;
Configurations[Tools.InspireLeftRH56DFX] = InspireLeftRH56DFXConfiguration;
Configurations[Tools.Robotiq_2F85] = Robotiq2F85Configuration;
Configurations[Tools.RobotiqHandE] = RobotiqHandEConfiguration;
Configurations[Tools.UnitreeRightHand] = UnitreeRightHandConfiguration;
Configurations[Tools.UnitreeLeftHand] = UnitreeLeftHandConfiguration;
Configurations[Tools.UnitreeRightDex3_1] = UnitreeRightDex31Configuration;
Configurations[Tools.UnitreeLeftDex3_1] = UnitreeLeftDex31Configuration;

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */



/* Download all the files needed to simulate and display the Franka Emika Panda robot

The files are stored in Mujoco's filesystem at '/scenes/franka_emika/panda'
*/
async function downloadPandaRobot() {
    const dstFolder = '/scenes/franka_emika/panda';
    const srcURL = getURL('models/franka_emika/panda/');

    await downloadScene(srcURL + 'panda.xml', dstFolder);
}


/* Download all the files needed to simulate and display the Franka Hand tool

The files are stored in Mujoco's filesystem at '/scenes/franka_emika/hand'
*/
async function downloadFrankHandTool() {
    const dstFolder = '/scenes/franka_emika/hand';
    const srcURL = getURL('models/franka_emika/hand/');

    await downloadScene(srcURL + 'hand.xml', dstFolder);
}


/* Download all the files needed to simulate and display the Unitree G1 robot

The files are stored in Mujoco's filesystem at '/scenes/unitree/g1'
*/
async function downloadG1Robot() {
    const dstFolder = '/scenes/unitree/g1';
    const srcURL = getURL('models/unitree/g1/');

    await downloadScene(srcURL + 'g1.xml', dstFolder);
    await downloadScene(srcURL + 'g1_fixed_legs.xml', dstFolder);
}


/* Download all the files needed to simulate and display the right Inspire RH56DFX tool

The files are stored in Mujoco's filesystem at '/scenes/inspire_robots/RH56DFX'
*/
async function downloadInspireRightRH56DFXTool() {
    const dstFolder = '/scenes/inspire_robots/RH56DFX';
    const srcURL = getURL('models/inspire_robots/RH56DFX/');

    await downloadScene(srcURL + 'right.xml', dstFolder);
}


/* Download all the files needed to simulate and display the left Inspire RH56DFX tool

The files are stored in Mujoco's filesystem at '/scenes/inspire_robots/RH56DFX'
*/
async function downloadInspireLeftRH56DFXTool() {
    const dstFolder = '/scenes/inspire_robots/RH56DFX';
    const srcURL = getURL('models/inspire_robots/RH56DFX/');

    await downloadScene(srcURL + 'left.xml', dstFolder);
}


/* Download all the files needed to simulate and display the Robotiq 2F85 tool

The files are stored in Mujoco's filesystem at '/scenes/robotiq/2f85'
*/
async function downloadRobotiq2F85Tool() {
    const dstFolder = '/scenes/robotiq/2f85';
    const srcURL = getURL('models/robotiq/2f85/');

    await downloadScene(srcURL + '2f85.xml', dstFolder);
}


/* Download all the files needed to simulate and display the Robotiq Hand-e tool

The files are stored in Mujoco's filesystem at '/scenes/robotiq/hand-e'
*/
async function downloadRobotiqHandETool() {
    const dstFolder = '/scenes/robotiq/hand-e';
    const srcURL = getURL('models/robotiq/hand-e/');

    await downloadScene(srcURL + 'hande.xml', dstFolder);
}


/* Download all the files needed to simulate and display the Universel Robots UR5 robot

The files are stored in Mujoco's filesystem at '/scenes/universal_robots_ur5e'
*/
async function downloadUR5Robot() {
    const dstFolder = '/scenes/universal_robots/ur5e';
    const srcURL = getURL('models/universal_robots/ur5e/');

    await downloadScene(srcURL + 'ur5e.xml', dstFolder);
}


/* Download all the files needed to simulate and display the right Unitree hand tool

The files are stored in Mujoco's filesystem at '/scenes/unitree/hands'
*/
async function downloadUnitreeRightHandTool() {
    const dstFolder = '/scenes/unitree/hands';
    const srcURL = getURL('models/unitree/hands/');

    await downloadScene(srcURL + 'right.xml', dstFolder);
}


/* Download all the files needed to simulate and display the left Unitree hand tool

The files are stored in Mujoco's filesystem at '/scenes/unitree/hands'
*/
async function downloadUnitreeLeftHandTool() {
    const dstFolder = '/scenes/unitree/hands';
    const srcURL = getURL('models/unitree/hands/');

    await downloadScene(srcURL + 'left.xml', dstFolder);
}


/* Download all the files needed to simulate and display the right Unitree Dex3-1 tool

The files are stored in Mujoco's filesystem at '/scenes/unitree/dex3_1'
*/
async function downloadUnitreeRightDex31Tool() {
    const dstFolder = '/scenes/unitree/dex3_1';
    const srcURL = getURL('models/unitree/dex3_1/');

    await downloadScene(srcURL + 'right.xml', dstFolder);
}


/* Download all the files needed to simulate and display the left Unitree Dex3-1 tool

The files are stored in Mujoco's filesystem at '/scenes/unitree/dex3_1'
*/
async function downloadUnitreeLeftDex31Tool() {
    const dstFolder = '/scenes/unitree/dex3_1';
    const srcURL = getURL('models/unitree/dex3_1/');

    await downloadScene(srcURL + 'left.xml', dstFolder);
}


/* Download all the files needed to simulate and display a robot
*/
async function downloadRobot(name) {
    if (typeof(name) == 'symbol')
        name = name.description;
 
    switch (name) {
        case Robots.Panda.description: await downloadPandaRobot(); return;
        case Robots.G1.description: await downloadG1Robot(); return;
        case Robots.G1_FixedLegs.description: await downloadG1Robot(); return;
        case Robots.UR5.description: await downloadUR5Robot(); return;

        default:
            console.error("Failed to download unknown robot: " + name);
    }
}


/* Download all the files needed to simulate and display a tool
*/
async function downloadTool(name) {
    if (typeof(name) == 'symbol')
        name = name.description;
 
    switch (name) {
        case Tools.FrankaHand.description: await downloadFrankHandTool(); return;
        case Tools.InspireRightRH56DFX.description: await downloadInspireRightRH56DFXTool(); return;
        case Tools.InspireLeftRH56DFX.description: await downloadInspireLeftRH56DFXTool(); return;
        case Tools.Robotiq_2F85.description: await downloadRobotiq2F85Tool(); return;
        case Tools.RobotiqHandE.description: await downloadRobotiqHandETool(); return;
        case Tools.UnitreeRightHand.description: await downloadUnitreeRightHandTool(); return;
        case Tools.UnitreeLeftHand.description: await downloadUnitreeLeftHandTool(); return;
        case Tools.UnitreeRightDex3_1.description: await downloadUnitreeRightDex31Tool(); return;
        case Tools.UnitreeLeftDex3_1.description: await downloadUnitreeLeftDex31Tool(); return;

        default:
            console.error("Failed to download unknown tool: " + name);
    }
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
                "link" + n,
                f, R, n,
                (n > 0) && (n < this.q.length) ? "joint" + n : null,
            );
            parent.append(xmlBody);

            if ((n > 0) && (n < this.q.length)) {
                const xmlGeneral = xmlDoc.createElement("general");
                xmlGeneral.setAttribute("name", "actuator" + n);
                xmlGeneral.setAttribute("joint", "joint" + n);
                xmlGeneral.setAttribute("gainprm", "4500");
                xmlGeneral.setAttribute("biasprm", "0 -4500 -450");
                xmlActuators.append(xmlGeneral);
            }

            parent = xmlBody;
        }

        const xmlSite = xmlDoc.createElement("site");
        xmlSite.setAttribute("name", "tcp");
        xmlSite.setAttribute("pos", "0 0 0");
        xmlSite.setAttribute("size", "0.001");
        parent.append(xmlSite);

        return xmlDoc;
    }

    getConfiguration() {
        const config = new RobotConfiguration();

        config.robotRoot = "link0";

        config.defaultPose = {};

        config.defaultToolConfigurations = [
            new DefaultRobotBuilderToolConfiguration
        ];

        for (let i = 0; i < this.defaultPose.length; ++i)
            config.defaultPose["joint" + (i + 1)] = this.defaultPose[i];

        return config;
    }

    _createDefaults(xmlDoc) {
        const xmlDefaults = xmlDoc.createElement("default");

        const xmlVisualDefault = xmlDoc.createElement("default");
        xmlVisualDefault.setAttribute("class", "visual");
        xmlDefaults.append(xmlVisualDefault);

        const xmlVisualGeomDefault = xmlDoc.createElement("geom");
        xmlVisualGeomDefault.setAttribute("contype", "0");
        xmlVisualGeomDefault.setAttribute("conaffinity", "0");
        xmlVisualGeomDefault.setAttribute("group", "2");
        xmlVisualDefault.append(xmlVisualGeomDefault);

        const xmlCollisionDefault = xmlDoc.createElement("default");
        xmlCollisionDefault.setAttribute("class", "collision");
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
            xmlGeom.setAttribute("class", "visual");
            xmlGeom.setAttribute("rgba", color);
            xmlBody.append(xmlGeom);

            xmlGeom = xmlDoc.createElement("geom");
            xmlGeom.setAttribute("type", "sphere");
            xmlGeom.setAttribute("size", "0.01");
            xmlGeom.setAttribute("class", "collision");
            xmlBody.append(xmlGeom);
        } else {
            let xmlGeom = xmlDoc.createElement("geom");
            xmlGeom.setAttribute("type", "sphere");
            xmlGeom.setAttribute("size", "0.02");
            xmlGeom.setAttribute("mass", ".1");
            xmlGeom.setAttribute("class", "visual");
            xmlGeom.setAttribute("rgba", color);
            xmlBody.append(xmlGeom);

            xmlGeom = xmlDoc.createElement("geom");
            xmlGeom.setAttribute("type", "sphere");
            xmlGeom.setAttribute("size", "0.02");
            xmlGeom.setAttribute("class", "collision");
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
                xmlGeom.setAttribute("class", "visual");
                xmlGeom.setAttribute("rgba", color);
                xmlBody.append(xmlGeom);

                const p1 = math.multiply(math.divide(p, math.norm(p)), 0.01);
                const p2 = math.subtract(p, p1);

                xmlGeom = xmlDoc.createElement("geom");
                xmlGeom.setAttribute("type", "cylinder");
                xmlGeom.setAttribute("size", "0.01");
                xmlGeom.setAttribute("fromto", p1.join(" ") + " " + p2.join(" "));
                xmlGeom.setAttribute("class", "collision");
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


class DefaultRobotBuilderToolConfiguration extends ToolConfiguration {
    constructor() {
        super();

        this.tcpSite = "tcp";
        this.tcpSize = 0.04;
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

/*
 * SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 */



const Bodies = Object.freeze({
    Box: Symbol("box"),
    Capsule: Symbol("capsule"),
    Cylinder: Symbol("cylinder"),
    Ellipsoid: Symbol("ellipsoid"),
    Plane: Symbol("plane"),
    Sphere: Symbol("sphere"),
    Mesh: Symbol("mesh"),
});


class BaseInfos {

    constructor(name, position, orientation) {
        this.name = name;
        this.position = position;
        this.orientation = orientation;

        if (this.position == null)
            this.position = new THREE.Vector3(0.0, 0.0, 0.0);
        else if (Array.isArray(this.position))
            this.position = new THREE.Vector3(this.position[0], this.position[1], this.position[2]);

        if (this.orientation == null)
            this.orientation = new THREE.Quaternion(0.0, 0.0, 0.0, 1.0);
        else if (Array.isArray(this.orientation))
            this.orientation = new THREE.Quaternion(this.orientation[0], this.orientation[1], this.orientation[2], this.orientation[3]);
    }

}


class RobotInfos extends BaseInfos {

    constructor(name, configuration, position, orientation, parameters, declared=false) {
        super(name, position, orientation);

        this.configuration = configuration;
        this.parameters = parameters;
        this.declared = declared;
        this.filename = null;

        // Ensure the parameters have default values
        if (this.parameters == null)
            this.parameters = new Map();
        else if (!(this.parameters instanceof Map))
            this.parameters = new Map(Object.entries(this.parameters));

        const defaults = new Map([
            ['use_toon_shader', false],
            ['use_light_toon_shader', false],
            ['hue', null],
            ['color', null],
            ['controlsEnabled', true],
            ['collisionsEnabled', true],
            ['layer', Layers.Base],
        ]);

        this.parameters = new Map([...defaults, ...this.parameters]);
    }

}


class PartInfos extends BaseInfos {

    constructor(name, filename, position, orientation, body) {
        super(name, position, orientation);

        this.filename = filename;
        this.body = body;
    }

}


class BodyInfos extends BaseInfos {

    constructor(type, name, position, orientation, color, mass) {
        super(name, position, orientation);

        this.type = type;
        this.color = color;
        this.mass = mass;

        if (this.color.length == 3)
            this.color = [this.color[0], this.color[1], this.color[2], 1.0];
    }

}


class BoxInfos extends BodyInfos {

    constructor(name, position, orientation, halfSize, color, mass) {
        super(Bodies.Box, name, position, orientation, color, mass);

        this.size = halfSize;

        if (Array.isArray(this.size))
            this.size = new THREE.Vector3(this.size[0], this.size[1], this.size[2]);
    }

}


class CapsuleInfos extends BodyInfos {

    constructor(name, position, orientation, radius, halfLength, color, mass) {
        super(Bodies.Capsule, name, position, orientation, color, mass);

        this.radius = radius;
        this.halfLength = halfLength;
    }

}


class CylinderInfos extends BodyInfos {

    constructor(name, position, orientation, radius, halfLength, color, mass) {
        super(Bodies.Cylinder, name, position, orientation, color, mass);

        this.radius = radius;
        this.halfLength = halfLength;
    }

}


class EllipsoidInfos extends BodyInfos {

    constructor(name, position, orientation, radiuses, color, mass) {
        super(Bodies.Ellipsoid, name, position, orientation, color, mass);

        this.radiuses = radiuses;

        if (Array.isArray(this.radiuses))
            this.radiuses = new THREE.Vector3(this.radiuses[0], this.radiuses[1], this.radiuses[2]);
    }

}


class PlaneInfos extends BodyInfos {

    constructor(name, position, orientation, halfSizeX, halfSizeY, spacing, color, mass) {
        super(Bodies.Plane, name, position, orientation, color, mass);

        this.halfSizeX = halfSizeX;
        this.halfSizeY = halfSizeY;
        this.spacing = spacing;
    }

}


class SphereInfos extends BodyInfos {

    constructor(name, position, orientation, radius, color, mass) {
        super(Bodies.Sphere, name, position, orientation, color, mass);
        this.radius = radius;
    }

}


class MeshInfos extends BodyInfos {

    constructor(name, position, orientation, filename, color, mass, meshPosition, meshOrientation, collision) {
        super(Bodies.Mesh, name, position, orientation, color, mass);
        this.filename = filename;
        this.meshPosition = meshPosition;
        this.meshOrientation = meshOrientation;

        if (this.meshPosition == null)
            this.meshPosition = new THREE.Vector3(0.0, 0.0, 0.0);
        else if (Array.isArray(this.meshPosition))
            this.meshPosition = new THREE.Vector3(this.meshPosition[0], this.meshPosition[1], this.meshPosition[2]);

        if (this.meshOrientation == null)
            this.meshOrientation = new THREE.Quaternion(0.0, 0.0, 0.0, 1.0);
        else if (Array.isArray(this.meshOrientation))
            this.meshOrientation = new THREE.Quaternion(this.meshOrientation[0], this.meshOrientation[1], this.meshOrientation[2], this.meshOrientation[3]);

        if (collision != null) {
            if (collision instanceof Array) {
                this.collision = [];
                for (let infos of collision) {
                    if (infos.type == 'box') {
                        this.collision.push(
                            new BoxInfos(null, infos.position, infos.orientation, infos.halfSize, [0, 0, 0])
                        );
                    }
                    else if (infos.type == 'capsule') {
                        this.collision.push(
                            new CapsuleInfos(
                                null, infos.position, infos.orientation, infos.radius, infos.halfLength, [0, 0, 0]
                            )
                        );
                    }
                    else if (infos.type == 'cylinder') {
                        this.collision.push(
                            new CylinderInfos(
                                null, infos.position, infos.orientation, infos.radius, infos.halfLength, [0, 0, 0]
                            )
                        );
                    }
                    else if (infos.type == 'ellipsoid') {
                        this.collision.push(
                            new EllipsoidInfos(
                                null, infos.position, infos.orientation, infos.radiuses, [0, 0, 0]
                            )
                        );
                    }
                    else if (infos.type == 'sphere') {
                        this.collision.push(
                            new SphereInfos(
                                null, infos.position, infos.orientation, infos.radius, [0, 0, 0]
                            )
                        );
                    }
                }
            } else {
                this.collision = collision;
            }
        } else {
            this.collision = true;
        }
    }

}


/**
 * High-level scene builder to assemble MuJoCo scenes programmatically and
 * generate processed XML files in the MuJoCo filesystem under `/scenes/...`.
 */
class SceneBuilder {

    /**
     * @param {string} scenePath - path to the base XML scene file
     */
    constructor(scenePath) {
        this.rootScenePath = new Path$1(scenePath);
        this.robots = [];
        this.bodies = [];
        this.parts = [];

        let xmlDoc = readXmlFile(this.rootScenePath.toString());
        if (xmlDoc == null)
        {
            console.error("Failed to load the file '" + this.rootScenePath.toString() + "'");
            return;
        }

        this.camera = getFreeCameraSettings(xmlDoc);
    }

    /**
     * Add a robot specification to the scene builder.
     * @param {string} name - robot name
     * @param {Object|string} configuration - robot configuration object or path
     * @param {THREE.Vector3|Array<number>|null} [position=null]
     * @param {THREE.Quaternion|Array<number>|null} [orientation=null]
     * @param {Object|null} [toolsConfiguration=null]
     * @param {Object|null} [parameters=null]
     * @returns {void}
     */
    addRobot(name, configuration, position=null, orientation=null, toolsConfiguration=null, parameters=null) {
        configuration = processConfiguration(configuration, toolsConfiguration);
        this.robots.push(new RobotInfos(name, configuration, position, orientation, parameters));
    }

    /**
     * Declare a robot already existing in the Mujoco scene.
     * @param {string} name
     * @param {Object|string} configuration
     * @param {Object|null} [toolsConfiguration=null]
     * @param {Object|null} [parameters=null]
     * @returns {void}
     */
    declareRobot(name, configuration, toolsConfiguration=null, parameters=null) {
        configuration = processConfiguration(configuration, toolsConfiguration);
        this.robots.push(new RobotInfos(name, configuration, null, null, parameters, true));
    }

    /**
     * Build a robot XML using a RobotBuilder and add it to the scene.
     * The generated XML is stored under /scenes/generated/.
     * @param {string} name
     * @param {RobotBuilder} robotBuilder
     * @param {THREE.Vector3|Array<number>|null} [position=null]
     * @param {THREE.Quaternion|Array<number>|null} [orientation=null]
     * @returns {void}
     */
    buildRobot(name, robotBuilder, position=null, orientation=null) {
        const folder = new Path$1('/scenes/generated/');
        const filename = folder.join(robotBuilder.name + '.xml');

        // Check that the folder exists
        if (!folder.exists())
            folder.mkdir();

        // Generate the XML file of the robot if needed
        if (!filename.exists()) {
            const xmlRobotDoc = robotBuilder.generateXMLDocument();

            const serializer = new XMLSerializer();
            filename.write(serializer.serializeToString(xmlRobotDoc));
        }

        // Add the robot to the scene
        const configuration = robotBuilder.getConfiguration();
        configuration.filename = filename.relativeTo('/scenes/').toString();
        this.addRobot(name, configuration, position, orientation);
    }

    /**
     * Add a box collision/visual body to the scene builder.
     * @param {string} name
     * @param {THREE.Vector3|Array<number>} position
     * @param {THREE.Quaternion|Array<number>} orientation
     * @param {THREE.Vector3|Array<number>} halfSize
     * @param {Array<number>} color - rgba or rgb
     * @param {number|null} [mass=null] If null, the body isn't movable
     * @returns {void}
     */
    addBox(name, position, orientation, halfSize, color, mass=null) {
        this.bodies.push(new BoxInfos(name, position, orientation, halfSize, color, mass));
    }

    /**
     * Add a capsule body.
     */
    addCapsule(name, position, orientation, radius, halfLength, color, mass=null) {
        this.bodies.push(new CapsuleInfos(name, position, orientation, radius, halfLength, color, mass));
    }

    /**
     * Add a cylinder body.
     */
    addCylinder(name, position, orientation, radius, halfLength, color, mass=null) {
        this.bodies.push(new CylinderInfos(name, position, orientation, radius, halfLength, color, mass));
    }

    /**
     * Add an ellipsoid body.
     */
    addEllipsoid(name, position, orientation, radiuses, color, mass=null) {
        this.bodies.push(new EllipsoidInfos(name, position, orientation, radiuses, color, mass));
    }

    /**
     * Add a plane body.
     */
    addPlane(name, position, orientation, halfSizeX, halfSizeY, spacing, color) {
        this.bodies.push(
            new PlaneInfos(name, position, orientation, halfSizeX, halfSizeY, spacing, color, null)
        );
    }

    /**
     * Add a sphere body.
     */
    addSphere(name, position, orientation, radius, color, mass=null) {
        this.bodies.push(new SphereInfos(name, position, orientation, radius, color, mass));
    }

    /**
     * Add a mesh body referencing an external mesh file.
     */
    addMesh(
        name, position, orientation, filename, color, mass=null, meshPosition=null, meshOrientation=null,
        collision=null
    ) {
        this.bodies.push(
            new MeshInfos(
                name, position, orientation, filename, color, mass, meshPosition, meshOrientation,
                collision
            )
        );
    }

    /**
     * Add a part defined in an external XML file and optionally reference a body name.
     * @param {string} name
     * @param {string} filename
     * @param {THREE.Vector3|Array<number>|null} [position=null]
     * @param {THREE.Quaternion|Array<number>|null} [orientation=null]
     * @param {string|Object|null} [body=null]
     * @returns {void}
     */
    addPart(name, filename, position=null, orientation=null, body=null) {
        if (body == null) {
            // Load the XML document
            let xmlDoc = readXmlFile(filename);
            if (xmlDoc == null)
            {
                console.error("Failed to load the file '" + filename + "'");
                return;
            }

            // Retrieve the first body
            let xmlBody = getFirstElementByTag(xmlDoc, 'body');
            if (xmlBody == null)
                return;

            body = xmlBody.getAttribute('name');
        }

        this.parts.push(new PartInfos(name, filename, position, orientation, body));
    }


    /**
     * Build the scene: process XML, copy assets into the MuJoCo filesystem and
     * generate a processed scene XML under `/scenes/<id>/<id>.xml`.
     * @returns {[string, Object]|[null,null]} tuple of generated filename and light intensities
     */
    build() {
        // Load the XML document
        let xmlDoc = readXmlFile(this.rootScenePath.toString());
        if (xmlDoc == null)
        {
            console.error("Failed to load the file '" + this.rootScenePath.toString() + "'");
            return [null, null];
        }

        // Generate the name of the processed file
        const uniqueId = getUniqueId(36);

        const rootDestFolder = new Path$1('/scenes/' + uniqueId);
        const subFolder = this.rootScenePath.dirname().relativeTo('/scenes/');

        const folder = rootDestFolder.join(subFolder);
        folder.mkdir();

        const filename = folder.join(uniqueId + '.xml');

        // Retrieve needed elements from the XML document
        const xmlRoot = getFirstElementByTag(xmlDoc, "mujoco");
        const xmlWorldBody = getFirstElementByTag(xmlDoc, 'worldbody');

        let xmlOption = getFirstElementByTag(xmlDoc, 'option');
        const options = {};

        if (xmlOption != null) {
            for (let name of xmlOption.getAttributeNames())
                options[name] = xmlOption.getAttribute(name);
        }

        // Complete or create the camera settings
        completeCameraSettings(xmlDoc, this.camera);

        // Retrieve or create the <asset> element
        let xmlAssets = getFirstElementByTag(xmlDoc, 'asset');
        if (xmlAssets == null) {
            xmlAssets = xmlDoc.createElement('asset');
            xmlRoot.insertBefore(xmlAssets, xmlWorldBody);
        }

        const lightIntensities = processLights(xmlWorldBody);

        // Process the included files
        const additionalLightIntensities = processIncludedFiles(xmlDoc, this.rootScenePath.dirname(), rootDestFolder);

        for (let name in additionalLightIntensities)
            lightIntensities[name] = additionalLightIntensities[name];

        // Add the robots
        for (let i = 0; i < this.robots.length; ++i) {
            const additionalLightIntensities = addRobot(
                this.robots[i], i, xmlDoc, xmlWorldBody, xmlAssets, options, rootDestFolder
            );

            for (let name in additionalLightIntensities)
                lightIntensities[name] = additionalLightIntensities[name];
        }

        // Add the parts
        for (let i = 0; i < this.parts.length; ++i)
            addPart(this.parts[i], i, xmlDoc, xmlWorldBody, xmlAssets);

        // Add the bodies
        for (let i = 0; i < this.bodies.length; ++i)
            addBody(this.bodies[i], xmlDoc, xmlWorldBody);
        
        // Put the merged options in the XML file
        if (Object.keys(options).length > 0) {
            if (xmlOption == null) {
                xmlOption = xmlDoc.createElement('option');
                xmlRoot.insertBefore(xmlOption, xmlRoot.children[0]);
            }

            for (let name in options)
                xmlOption.setAttribute(name, options[name]);
        }

        // Adapt the 'meshdir' and 'texturedir' settings
        let xmlCompiler = getOrCreateCompiler(xmlDoc);

        xmlCompiler.setAttribute(
            'meshdir',
            this.rootScenePath.dirname().join(xmlCompiler.getAttribute('meshdir')).toString()
        );

        xmlCompiler.setAttribute(
            'texturedir',
            this.rootScenePath.dirname().join(xmlCompiler.getAttribute('texturedir')).toString()
        );

        // Save the scene XML file
        const serializer = new XMLSerializer();
        filename.write(serializer.serializeToString(xmlDoc));

        return [filename.toString(), lightIntensities];
    }

}


function processConfiguration(configuration, toolsConfiguration) {
    if (typeof(configuration) == 'string')
        configuration = Robots[configuration];

    if (typeof(configuration) == 'symbol')
        configuration = new Configurations[configuration]();

    if (toolsConfiguration != null) {
        if (!Array.isArray(toolsConfiguration))
            toolsConfiguration = [toolsConfiguration];
    } else {
        toolsConfiguration = [];
    }

    while (toolsConfiguration.length < configuration.defaultToolConfigurations.length)
        toolsConfiguration.push(null);

    for (let i = 0; i < toolsConfiguration.length; ++i) {
        if (toolsConfiguration[i] == null)
            toolsConfiguration[i] = configuration.defaultToolConfigurations[i];

        else if (typeof(toolsConfiguration[i]) == 'string')
            toolsConfiguration[i] = Tools[toolsConfiguration[i]];

        if (typeof(toolsConfiguration[i]) == 'symbol')
            toolsConfiguration[i] = new Configurations[toolsConfiguration[i]]();
    }

    configuration.tools = toolsConfiguration;

    return configuration;
}


function processIncludedFiles(xmlDoc, rootSrcFolder, rootDestFolder) {
    const serializer = new XMLSerializer();

    const lightIntensities = {};

    // Search for include directives
    const xmlIncludes = xmlDoc.getElementsByTagName("include");
    for (let xmlInclude of xmlIncludes) {
        let includedFile = new Path$1(xmlInclude.getAttribute("file"));

        // Generate the name of the processed file
        const uniqueId = getUniqueId(36);

        const subFolder = rootSrcFolder.relativeTo('/scenes/');

        const dstFolder = includedFile.isAbsolute ? includedFile.dirname() : rootDestFolder.join(subFolder, includedFile.dirname());
        dstFolder.mkdir();

        const dstFilename = dstFolder.join(uniqueId + '.xml');

        // Load the XML document
        const srcFilename = includedFile.isAbsolute ? includedFile : rootSrcFolder.join(includedFile);
        const srcFolder = srcFilename.dirname();

        let xmlIncludedDoc = readXmlFile(srcFilename.toString());
        if (xmlIncludedDoc == null) {
            console.error("Failed to load the file '" + srcFilename.toString() + "'");
            continue;
        }

        // Adapt the 'meshdir' and 'texturedir' settings
        let xmlCompiler = getOrCreateCompiler(xmlIncludedDoc);

        xmlCompiler.setAttribute(
            'meshdir',
            srcFolder.join(xmlCompiler.getAttribute('meshdir')).toString()
        );

        xmlCompiler.setAttribute(
            'texturedir',
            srcFolder.join(xmlCompiler.getAttribute('texturedir')).toString()
        );

        // Process the lights
        let additionalLightIntensities = processLights(xmlIncludedDoc);
        for (let name in additionalLightIntensities)
            lightIntensities[name] = additionalLightIntensities[name];

        // Process the included files in that file
        additionalLightIntensities = processIncludedFiles(xmlIncludedDoc, rootSrcFolder, rootDestFolder);
        for (let name in additionalLightIntensities)
            lightIntensities[name] = additionalLightIntensities[name];

        dstFilename.write(serializer.serializeToString(xmlIncludedDoc));

        xmlInclude.setAttribute("file", dstFilename.toString());
    }

    return lightIntensities;
}


function processLights(xmlWorldBody) {
    let intensities = {};

    const xmlLights = Array.from(xmlWorldBody.getElementsByTagName('light'));
    for (let xmlLight of xmlLights) {
        let intensity = xmlLight.getAttribute('intensity');
        if (intensity != null) {
            intensity = Number.parseFloat(intensity);
            xmlLight.removeAttribute('intensity');
        } else {
            intensity = 1;
        }

        let name = xmlLight.getAttribute('name');
        if (name == null) {
            name = '__v3d_light_' + getUniqueId(12);
            xmlLight.setAttribute('name', name);
        }

        intensities[name] = intensity;
    }

    return intensities;
}


function addRobot(robot, index, xmlDoc, xmlWorldBody, xmlAssets, options, rootDestFolder) {
    if (robot.declared)
        return;

    const prefix = robot.name + '_';

    let robotOptions = null;
    let lightIntensities = null;
    [robot.filename, robotOptions, lightIntensities] = processRobotXmlFile(robot, rootDestFolder);
    if (robot.filename == null)
        return;

    // Merge the options
    for (let name in robotOptions) {
        if (!(name in options))
            options[name] = robotOptions[name];
    }

    // Include the robot in the scene
    const xmlModel = xmlDoc.createElement('model');
    xmlModel.setAttribute('name', '_v3d_robot_' + index);
    xmlModel.setAttribute('file', robot.filename);
    xmlAssets.appendChild(xmlModel);

    const xmlAttach = xmlDoc.createElement('attach');
    xmlAttach.setAttribute('model', '_v3d_robot_' + index);
    xmlAttach.setAttribute('body', robot.configuration.robotRoot);
    xmlAttach.setAttribute('prefix', prefix);
    xmlWorldBody.appendChild(xmlAttach);

    // Update the prefixes in the configurations
    const prefixedTools = [];
    for (let j = 0; j < robot.configuration.tools.length; ++j) {
        const cfg = robot.configuration.tools[j];
        const toolPrefix = (cfg.filename != null ? prefix + 'tool_' + j + '_' : prefix);
        prefixedTools.push(cfg.addPrefix(toolPrefix));
    }

    robot.configuration = robot.configuration.addPrefix(prefix);
    robot.configuration.tools = prefixedTools;

    // Update the prefixes of the lights
    const prefixedLightIntensities = {};
    for (let name in lightIntensities)
        prefixedLightIntensities[prefix + name] = lightIntensities[name];

    return prefixedLightIntensities;
}


function addPart(part, index, xmlDoc, xmlWorldBody, xmlAssets) {
    const prefix = part.name + '_';

    // Include the part in the scene
    const xmlModel = xmlDoc.createElement('model');
    xmlModel.setAttribute('name', '_v3d_part_' + index);
    xmlModel.setAttribute('file', part.filename);
    xmlAssets.appendChild(xmlModel);

    const xmlBody = xmlDoc.createElement('body');
    xmlBody.setAttribute('name', '_v3d_part_' + index + '_frame');
    xmlBody.setAttribute('pos', '' + part.position.x + ' ' + part.position.y + ' ' + part.position.z);
    xmlBody.setAttribute(
        'quat',
        '' + part.orientation.w + ' ' + part.orientation.x + ' ' + part.orientation.y + ' ' +
        part.orientation.z
    );
    xmlWorldBody.appendChild(xmlBody);

    const xmlAttach = xmlDoc.createElement('attach');
    xmlAttach.setAttribute('model', '_v3d_part_' + index);
    xmlAttach.setAttribute('body', part.body);
    xmlAttach.setAttribute('prefix', prefix);
    xmlBody.appendChild(xmlAttach);
}


function addBody(body, xmlDoc, xmlWorldBody) {
    const xmlBody = xmlDoc.createElement('body');
    xmlBody.setAttribute('name', body.name);
    xmlBody.setAttribute('pos', '' + body.position.x + ' ' + body.position.y + ' ' + body.position.z);
    xmlBody.setAttribute(
        'quat',
        '' + body.orientation.w + ' ' + body.orientation.x + ' ' + body.orientation.y + ' ' +
        body.orientation.z
    );

    xmlWorldBody.appendChild(xmlBody);

    if ((body.mass != null) && (body.type != Bodies.Plane)) {
        const xmlFreeJoint = xmlDoc.createElement('freejoint');
        xmlBody.appendChild(xmlFreeJoint);
    }

    const xmlGeom = xmlDoc.createElement('geom');
    xmlGeom.setAttribute('type', body.type.description);
    xmlGeom.setAttribute('rgba', '' + body.color[0] + ' ' + body.color[1] + ' ' + body.color[2] + ' ' + body.color[3]);
    xmlBody.appendChild(xmlGeom);

    setupGeometry(xmlGeom, body);

    if (body.type == Bodies.Mesh) {
        xmlGeom.setAttribute('pos', '' + body.meshPosition.x + ' ' + body.meshPosition.y + ' ' + body.meshPosition.z);

        xmlGeom.setAttribute(
            'quat',
            '' + body.meshOrientation.w + ' ' + body.meshOrientation.x + ' ' + body.meshOrientation.y + ' ' +
            body.meshOrientation.z
        );

        let xmlAsset = getFirstElementByTag(xmlDoc, 'asset');
        if (xmlAsset == null) {
            xmlAsset = xmlDoc.createElement('asset');
            const xmlRoot = getFirstElementByTag(xmlDoc, 'mujoco');
            xmlRoot.insertBefore(xmlAsset, xmlWorldBody);
        }

        const xmlMesh = xmlDoc.createElement('mesh');
        xmlMesh.setAttribute('name', body.name);
        xmlMesh.setAttribute('file', body.filename);
        xmlAsset.appendChild(xmlMesh);

        if (body.collision !== true) {
            xmlGeom.setAttribute('contype', 0);
            xmlGeom.setAttribute('conaffinity', 0);
            xmlGeom.setAttribute('group', 2);
        }

        if (body.collision instanceof Array) {
            for (let infos of body.collision) {
                const xmlCollisionGeom = xmlDoc.createElement('geom');
                xmlCollisionGeom.setAttribute('type', infos.type.description);
                xmlCollisionGeom.setAttribute('group', 3);
                xmlBody.appendChild(xmlCollisionGeom);

                setupGeometry(xmlCollisionGeom, infos);

                if (infos.position != null) {
                    xmlCollisionGeom.setAttribute(
                        'pos',
                        '' + infos.position.x + ' ' + infos.position.y + ' ' + infos.position.z
                    );
                }

                if (body.orientation != null) {
                    xmlCollisionGeom.setAttribute(
                        'quat',
                        '' + infos.orientation.w + ' ' + infos.orientation.x + ' ' + infos.orientation.y + ' ' +
                        infos.orientation.z
                    );
                }
            }
        }
    }
}


function setupGeometry(xmlGeom, body) {
    if (body.mass != null)
        xmlGeom.setAttribute('mass', body.mass);

    if (body.type == Bodies.Box) {
        xmlGeom.setAttribute('size', '' + body.size.x + ' ' + body.size.y + ' ' + body.size.z);

        // xmlGeom.setAttribute('friction', "10 10 10");
        // xmlGeom.setAttribute('solimp', "0.95 0.99 0.001");
        // xmlGeom.setAttribute('solref', "0.004 1");
    }
    else if (body.type == Bodies.Capsule) {
        xmlGeom.setAttribute('size', '' + body.radius + ' ' + body.halfLength);
    }
    else if (body.type == Bodies.Cylinder) {
        xmlGeom.setAttribute('size', '' + body.radius + ' ' + body.halfLength);
    }
    else if (body.type == Bodies.Ellipsoid) {
        xmlGeom.setAttribute('size', '' + body.radiuses.x + ' ' + body.radiuses.y + ' ' + body.radiuses.z);
    }
    else if (body.type == Bodies.Plane) {
        xmlGeom.setAttribute('size', '' + body.halfSizeX + ' ' + body.halfSizeY + ' ' + body.spacing);
    }
    else if (body.type == Bodies.Sphere) {
        xmlGeom.setAttribute('size', '' + body.radius);
    }
    else if (body.type == Bodies.Mesh) {
        xmlGeom.setAttribute('mesh', body.name);
    }
}


function processRobotXmlFile(robot, rootDestFolder) {
    // Load the XML document
    let xmlRobotDoc = readXmlFile('/scenes/' + robot.configuration.filename);
    if (xmlRobotDoc == null)
    {
        console.error("Failed to load the file '" + robot.configuration.filename + "'");
        return [null, null, null];
    }

    // Retrieve the options
    let xmlOption = getFirstElementByTag(xmlRobotDoc, 'option');
    const options = {};

    if (xmlOption != null) {
        for (let name of xmlOption.getAttributeNames())
            options[name] = xmlOption.getAttribute(name);
    }

    // Retrieve the lights
    const lightIntensities = processLights(xmlRobotDoc);
    let hasLights = false;
    if (Object.keys(lightIntensities).length > 0) {
        for (let name in lightIntensities) {
            if (name.startsWith('__v3d_light_')) {
                hasLights = true;
                break;
            }
        }
    }

    // Check if there are no change to apply to the XML file, in which case we can use the original one
    const hasTransforms = !robot.position.equals(new THREE.Vector3(0.0, 0.0, 0.0)) ||
                          !robot.orientation.equals(new THREE.Quaternion(0.0, 0.0, 0.0));

    let hasTools = false;
    for (let i = 0; i < robot.configuration.tools.length; ++i)
        hasTools |= (robot.configuration.tools[i].filename != null);

    if (!hasTransforms && !hasTools && !hasLights && robot.parameters.get('collisionsEnabled')) {
        return ['/scenes/' + robot.configuration.filename, options, lightIntensities];
    }

    // Update the 'meshdir' and 'texturedir' settings
    const xmlCompiler = getFirstElementByTag(xmlRobotDoc, 'compiler');
    if (xmlCompiler != null)
    {
        const offset = robot.configuration.filename.lastIndexOf('/');
        xmlCompiler.setAttribute(
            'meshdir',
            '/scenes/' + robot.configuration.filename.substring(0, offset + 1) + xmlCompiler.getAttribute('meshdir')
        );
        xmlCompiler.setAttribute(
            'texturedir',
            '/scenes/' + robot.configuration.filename.substring(0, offset + 1) + xmlCompiler.getAttribute('texturedir')
        );
    }

    // Retrieve or create the <assets> element
    let xmlAssets = getFirstElementByTag(xmlRobotDoc, 'asset');
    const xmlRoot = getFirstElementByTag(xmlRobotDoc, 'mujoco');
    const xmlWorldBody = getFirstElementByTag(xmlRobotDoc, 'worldbody');
    if (xmlAssets == null) {
        xmlAssets = xmlRobotDoc.createElement('asset');
        xmlRoot.insertBefore(xmlAssets, xmlWorldBody);
    }

    // Update the transforms (if necessary)
    if (hasTransforms) {
        const xmlBody = getElementByTagAndName(xmlWorldBody, 'body', robot.configuration.robotRoot);

        const position = new THREE.Vector3(0.0, 0.0, 0.0);

        const xmlPosition = xmlBody.getAttribute('pos');
        if (xmlPosition) {
            const parts = xmlPosition.split(' ');
            position.x = Number.parseFloat(parts[0]);
            position.y = Number.parseFloat(parts[1]);
            position.z = Number.parseFloat(parts[2]);
        }

        position.add(robot.position);

        xmlBody.setAttribute('pos', '' + position.x + ' ' + position.y + ' ' + position.z);

        const orientation = new THREE.Quaternion(0.0, 0.0, 0.0, 1.0);

        const xmlQuat = xmlBody.getAttribute('quat');
        if (xmlQuat) {
            const parts = xmlQuat.split(' ');
            orientation.w = Number.parseFloat(parts[0]);
            orientation.x = Number.parseFloat(parts[1]);
            orientation.y = Number.parseFloat(parts[2]);
            orientation.z = Number.parseFloat(parts[3]);
        }

        orientation.multiply(robot.orientation);

        xmlBody.setAttribute(
            'quat',
            '' + orientation.w + ' ' + orientation.x + ' ' + orientation.y + ' ' + orientation.z
        );
    }

    // Disable the collisions (if necessary)
    if (!robot.parameters.get('collisionsEnabled')) {
        const xmlGeoms = Array.from(xmlWorldBody.getElementsByTagName("geom"));
        for (let xmlGeom of xmlGeoms) {
            if (xmlGeom.getAttribute("class").search("collision") >= 0)
                xmlGeom.parentElement.removeChild(xmlGeom);
        }
    }

    // Generate unique filename
    const uniqueId = getUniqueId(36);

    const srcFilename = new Path$1(robot.configuration.filename);
    const subFolder = srcFilename.dirname();

    const dstFolder = rootDestFolder.join(subFolder);
    dstFolder.mkdir();

    const dstFilename = dstFolder.join(uniqueId + '.xml');

    // Process the tools
    for (let j = 0; j < robot.configuration.tools.length; ++j) {
        const cfg = robot.configuration.tools[j];

        if (cfg.filename == null)
            continue;

        const prefix = 'tool_' + j + '_';

        // Process the XML file of the tool
        let toolOptions = null;
        let additionalLightIntensities = null;
        [cfg.filename, toolOptions, additionalLightIntensities] = processToolXmlFile(cfg, j, robot.parameters, rootDestFolder);
        if (cfg.filename == null)
            continue;

        // Merge the options
        for (let name in toolOptions) {
            if (!(name in options))
                options[name] = toolOptions[name];
        }

        // Merge the lights
        for (let name in additionalLightIntensities)
            lightIntensities[prefix + name] = additionalLightIntensities[name];

        // Include the tool in the robot
        const xmlModel = xmlRobotDoc.createElement('model');
        xmlModel.setAttribute('name', '_v3d_tool_' + j);
        xmlModel.setAttribute('file', cfg.filename);
        xmlAssets.appendChild(xmlModel);

        const xmlAttach = xmlRobotDoc.createElement('attach');
        xmlAttach.setAttribute('model', '_v3d_tool_' + j);
        xmlAttach.setAttribute('body', cfg.root);
        xmlAttach.setAttribute('prefix', prefix);

        const xmlSite = getElementByTagAndName(
            xmlRobotDoc, 'site', robot.configuration.defaultToolConfigurations[j].tcpSite
        );
        xmlSite.parentElement.insertBefore(xmlAttach, xmlSite);
    }

    // Save the robot XML file
    const serializer = new XMLSerializer();
    dstFilename.write(serializer.serializeToString(xmlRobotDoc));

    return [dstFilename.toString(), options, lightIntensities];
}


function processToolXmlFile(configuration, index, parameters, rootDestFolder) {
    // Load the XML document
    let xmlToolDoc = readXmlFile('/scenes/' + configuration.filename);
    if (xmlToolDoc == null)
    {
        console.error("Failed to load the file '" + configuration.filename + "'");
        return [null, null, null];
    }

    // Retrieve the options
    let xmlOption = getFirstElementByTag(xmlToolDoc, 'option');
    const options = {};

    if (xmlOption != null) {
        for (let name of xmlOption.getAttributeNames())
            options[name] = xmlOption.getAttribute(name);
    }

    // Retrieve the lights
    const lightIntensities = processLights(xmlToolDoc);
    let hasLights = false;
    if (Object.keys(lightIntensities).length > 0) {
        for (let name in lightIntensities) {
            if (name.startsWith('__v3d_light_')) {
                hasLights = true;
                break;
            }
        }
    }

    // Check if there are no change to apply to the XML file, in which case we can use the original one
    if (!hasLights && parameters.get('collisionsEnabled')) {
        return ['/scenes/' + configuration.filename, options, lightIntensities];
    }

    // Update the 'meshdir' and 'texturedir' settings
    const xmlCompiler = getFirstElementByTag(xmlToolDoc, 'compiler');
    if (xmlCompiler != null)
    {
        const offset = configuration.filename.lastIndexOf('/');
        xmlCompiler.setAttribute(
            'meshdir',
            '/scenes/' + configuration.filename.substring(0, offset + 1) + xmlCompiler.getAttribute('meshdir')
        );
        xmlCompiler.setAttribute(
            'texturedir',
            '/scenes/' + configuration.filename.substring(0, offset + 1) + xmlCompiler.getAttribute('texturedir')
        );
    }

    // Disable the collisions (if necessary)
    if (!parameters.get('collisionsEnabled')) {
        const xmlWorldBody = getFirstElementByTag(xmlToolDoc, 'worldbody');
        const xmlGeoms = Array.from(xmlWorldBody.getElementsByTagName("geom"));
        for (let xmlGeom of xmlGeoms) {
            if (xmlGeom.getAttribute("class").search("collision") >= 0)
                xmlGeom.parentElement.removeChild(xmlGeom);
        }
    }

    // Generate unique filename
    const uniqueId = getUniqueId(36);

    const srcFilename = new Path$1(configuration.filename);
    const subFolder = srcFilename.dirname();

    const dstFolder = rootDestFolder.join(subFolder);
    dstFolder.mkdir();

    const dstFilename = dstFolder.join(uniqueId + '.xml');

    // Save the tool XML file
    const serializer = new XMLSerializer();
    dstFilename.write(serializer.serializeToString(xmlToolDoc));

    return [dstFilename.toString(), options, lightIntensities];
}


function getOrCreateCompiler(xmlDoc) {
    let xmlCompiler = getFirstElementByTag(xmlDoc, 'compiler');
    if (xmlCompiler == null) {
        xmlCompiler = xmlDoc.createElement('compiler');
        const xmlMujoco = getFirstElementByTag(xmlDoc, 'mujoco');
        xmlMujoco.insertBefore(xmlCompiler, xmlMujoco.children[0]);
    }

    if (!xmlCompiler.hasAttribute('meshdir'))
        xmlCompiler.setAttribute('meshdir', '');

    if (!xmlCompiler.hasAttribute('texturedir'))
        xmlCompiler.setAttribute('texturedir', '');

    return xmlCompiler;
}


function completeCameraSettings(xmlDoc, settings) {
    let xmlVisual = getFirstElementByTag(xmlDoc, 'visual');
    if (xmlVisual == null) {
        xmlVisual = xmlDoc.createElement('visual');
        const xmlMujoco = getFirstElementByTag(xmlDoc, 'mujoco');
        xmlMujoco.insertBefore(xmlVisual, xmlMujoco.children[0]);
    }

    let xmlGlobal = getFirstElementByTag(xmlVisual, 'global');
    if (xmlGlobal == null) {
        xmlGlobal = xmlDoc.createElement('global');
        xmlVisual.append(xmlGlobal);
    }

    xmlGlobal.setAttribute('fovy', settings.fovy);
    xmlGlobal.setAttribute('azimuth', settings.azimuth);
    xmlGlobal.setAttribute('elevation', settings.elevation);

    let xmlMap = getFirstElementByTag(xmlVisual, "map");
    if (xmlMap == null) {
        xmlMap = xmlDoc.createElement('map');
        xmlVisual.append(xmlMap);
    }

    xmlMap.setAttribute('znear', settings.znear);
    xmlMap.setAttribute('zfar', settings.zfar);
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



/**
 * Entry point for the viewer3d.js library.
 *
 * Creates a three.js scene, camera, effect composer and ties the MuJoCo physics
 * simulator to rendering and user interaction.
 *
 * Usage:
 *   const v = new Viewer3D(domElement, parameters);
 *
 * Public surface (selected):
 * - constructor(domElement, parameters)
 * - setRenderingCallback(callback, timestep)
 * - loadScene(sceneBuilder)
 * - addTarget/removeTarget/getTarget
 * - addArrow/removeArrow/getArrow
 * - addPath/removePath/getPath
 * - addPoint/removePoint/getPoint
 * - addGaussian/removeGaussian/getGaussian
 * - enableControls / enableEndEffectorManipulation / enableJointsManipulation
 * - render()
 *
 * @class
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
        to add corresponding render passes.

        The default passes used are:

            - Passes.BaseRenderPass (LayerRenderPass): render objects in layer 'Layers.Base'
            - Passes.NoShadowsRenderPass (LayerRenderPass): render objects in layer 'Layers.NoShadows'
            - Passes.TopRenderPass (LayerRenderPass): render objects in layer 'Layers.Top'.
                                                      Note that the depth buffer is cleared by
                                                      this pass.
            - Passes.OutputPass (OutputPass): tone mapping, sRGB conversion

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
        this.axes = new ObjectList();
        this.physicalBodies = new ObjectList();

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
        this.renderingCallbackTimestep = -1;

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


    /**
     * Dispose of renderer resources. Call when the viewer is no longer needed.
     * @returns {void}
     */
    dispose() {
        this.renderer.dispose();
    }


    /**
     * Register a rendering callback called once per frame.
     * The callback can update joint positions or perform custom physics/logic.
     * Only one callback can be registered at a time.
     *
     * @param {(delta:number, time?:number)=>void|null} renderingCallback - function called each frame or null to unregister
     * @param {number} [timestep=-1.0] - fixed timestep for callback; if >0 the callback will be called repeatedly to catch up if necessary
     * @returns {void}
     */
    setRenderingCallback(renderingCallback, timestep=-1) {
        this.renderingCallback = renderingCallback;
        this.renderingCallbackTimestep = timestep;
        this.renderingCallbackTime = null;
    }


    /**
     * Set callbacks invoked when a user control interaction starts and ends.
     *
     * @param {(void)=>void|null} startCallback - called when control interaction begins
     * @param {(void)=>void|null} endCallback - called when control interaction ends
     * @returns {void}
     */
    setControlCallbacks(startCallback, endCallback) {
        this.controlStartedCallback = startCallback;
        this.controlEndedCallback = endCallback;
    }


    /**
     * Enable or disable manipulation controls (transform widgets).
     * Manipulation controls include end-effector and target manipulators.
     *
     * @param {boolean} enabled
     * @returns {void}
     */
    enableControls(enabled) {
        this.controlsEnabled = enabled;
        this.transformControls.enable(enabled);
        this.enableRobotTools(this.toolsEnabled);
    }


    /**
     * Returns whether manipulation controls are enabled.
     * @returns {boolean}
     */
    areControlsEnabled() {
        return this.controlsEnabled;
    }


    /**
     * Enable or disable manipulation of the end-effector (click to move TCP).
     * Note: control master flag (`controlsEnabled`) must also be true to allow manipulation.
     *
     * @param {boolean} enabled
     * @returns {void}
     */
    enableEndEffectorManipulation(enabled) {
        this.endEffectorManipulationEnabled = enabled && (this.planarIkControls != null);

        if (enabled) {
            for (let name in this.robots) {
                const robot = this.robots[name];
                if (robot.controlsEnabled && robot.tools.length > 0 && robot.tools[0].tcpTarget == null)
                    robot._createTcpTarget();
            }
        }
    }


    /**
     * Returns whether end-effector manipulation is enabled.
     * @returns {boolean}
     */
    isEndEffectorManipulationEnabled() {
        return this.endEffectorManipulationEnabled;
    }


    /**
     * Enable or disable joint manipulation (click/drag on joints or mouse wheel).
     *
     * @param {boolean} enabled
     * @returns {void}
     */
    enableJointsManipulation(enabled) {
        this.jointsManipulationEnabled = enabled;

        if ((this.interactionState == InteractionStates.JointHovering) ||
            (this.interactionState == InteractionStates.JointDisplacement) ||
            (this.interactionState == InteractionStates.LinkDisplacement)) {
                this._switchToInteractionState(InteractionStates.Default);
        }
    }


    /**
     * Returns whether joint manipulation is enabled.
     * @returns {boolean}
     */
    isJointsManipulationEnabled() {
        return this.jointsManipulationEnabled;
    }


    /**
     * Enable or disable link manipulation (drag links to move chains). Requires a planar IK
     * implementation to be available.
     *
     * @param {boolean} enabled
     * @returns {void}
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


    /**
     * Returns whether link manipulation is enabled.
     * @returns {boolean}
     */
    isLinksManipulationEnabled() {
        return this.linksManipulationEnabled;
    }


    /**
     * Enable or disable manipulation of scene objects (targets, gaussians, points).
     * @param {boolean} enabled
     * @returns {void}
     */
    enableObjectsManipulation(enabled) {
        this.objectsManipulationEnabled = enabled;
    }


    /**
     * Enable or disable application of force impulses to links when clicked.
     * @param {boolean} enabled
     * @param {number} [amount=0.0] - magnitude of applied impulses
     * @returns {void}
     */
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


    /**
     * Returns whether force impulses are enabled.
     * @returns {boolean}
     */
    areForceImpulsesEnabled() {
        return this.forceImpulsesEnabled;
    }


    /**
     * Returns whether object manipulation (targets/gaussians/points) is enabled.
     * @returns {boolean}
     */
    isObjectsManipulationEnabled() {
        return this.objectsManipulationEnabled;
    }


    /**
     * Enable or disable robot tools (grippers) for all robots.
     * @param {boolean} enabled
     * @returns {void}
     */
    enableRobotTools(enabled) {
        this.toolsEnabled = enabled;

        for (const name in this.robots)
            this.robots[name]._enableTools(this.toolsEnabled && this.controlsEnabled && this.robots[name].controlsEnabled);
    }


    /**
     * Returns whether robot tools (grippers) are enabled.
     * @returns {boolean}
     */
    areRobotToolsEnabled() {
        return this.toolsEnabled;
    }


    /**
     * Activate the layer where newly created objects will be placed.
     * @param {number} layer - layer index
     * @returns {void}
     */
    activateLayer(layer) {
        this.activeLayer = layer;

        this.passes[Passes.NoShadowsRenderPass].enabled = true;
        this.passes[Passes.TopRenderPass].enabled = true;
    }


    /**
     * Insert a render pass before a standard pass in the composer.
     * @param {Pass} pass - an EffectComposer pass instance
     * @param {symbol|number} standardPassId - an ID from Passes
     * @returns {void}
     */
    addPassBefore(pass, standardPassId) {
        let index = this.composer.passes.indexOf(this.passes[standardPassId]);
        this.composer.insertPass(pass, index);
    }


    /**
     * Insert a render pass after a standard pass in the composer.
     * @param {Pass} pass - an EffectComposer pass instance
     * @param {symbol|number} standardPassId - an ID from Passes
     * @returns {void}
     */
    addPassAfter(pass, standardPassId) {
        let index = this.composer.passes.indexOf(this.passes[standardPassId]);
        this.composer.insertPass(pass, index+1);
    }


    /**
     * Load a scene built by SceneBuilder or a filename string. Destroys previous
     * scene, loads MuJoCo assets, creates robots and configures camera/fog.
     *
     * @param {SceneBuilder|string} sceneBuilder
     * @returns {void}
     */
    loadScene(sceneBuilder) {
        // Destroy the previous scene (if any)
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

        // Load the scene
        if (typeof(sceneBuilder) == 'string')
            sceneBuilder = new SceneBuilder(sceneBuilder);

        const [filename, lightIntensities] = sceneBuilder.build();
        this.physicsSimulator = loadScene(filename, lightIntensities);
        this.scene.add(this.physicsSimulator.root);

        // Retrieve the scene statistics
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

        // Create the robots
        for (let i = 0; i < sceneBuilder.robots.length; ++i) {
            this._createRobot(sceneBuilder.robots[i]);
        }
    }


    _createRobot(infos) {
        if (this.physicsSimulator == null)
            return null;

        if (infos.name in this.robots)
            return null;

        const robot = this.physicsSimulator.createRobot(infos);
        if (robot == null)
            return;

        robot.controlsEnabled = infos.parameters.get('controlsEnabled');

        const toolEnabled = this.toolsEnabled && this.controlsEnabled && robot.controlsEnabled;

        const robotsWithControlsEnabled = Object.values(this.robots).filter((r) => r.controlsEnabled);

        if (toolEnabled && (robotsWithControlsEnabled.length == 1))
        {
            for (const name in this.robots) {
                this.robots[name]._enableTools(false);
                this.robots[name]._enableTools(true);
            }
        }

        if (toolEnabled && (robotsWithControlsEnabled.length == 0) && (infos.configuration.tools.length == 1) &&
            (infos.configuration.tools[0].root != null)) {
            robot.enableTool(toolEnabled, new GripperToolbarSection(this.toolbar, robot));
        } else {
            robot._enableTools(toolEnabled);
        }

        this.physicsSimulator.simulation.forward();
        this.physicsSimulator.synchronize();

        this.activateLayer(infos.parameters.get('layer'));

        robot.layers.disableAll();
        robot.layers.enable(this.activeLayer);

        this.robots[infos.name] = robot;

        if (this.parameters.get('show_joint_positions')) {
            robot.createJointPositionHelpers(
                this.scene, Layers.NoShadows, this.parameters.get('joint_position_colors')
            );

            this.passes[Passes.NoShadowsRenderPass].enabled = true;
            this.passes[Passes.TopRenderPass].enabled = true;
        }

        if (infos.parameters.get('color') != null) {
            let color = infos.parameters.get('color');
            color = new THREE.Color().setRGB(color[0], color[1], color[2], THREE.SRGBColorSpace);

            for (let segment of robot.segments)
                segment.visual.meshes.forEach((mesh) => modifyMaterialColor(mesh, color));

            for (let tool of robot.tools)
                tool.visual.meshes.forEach((mesh) => modifyMaterialColor(mesh, color));
        }

        if (infos.parameters.get('use_toon_shader')) {
            for (let segment of robot.segments)
                segment.visual.meshes.forEach((mesh) => enableToonShading(mesh));

            for (let tool of robot.tools)
                tool.visual.meshes.forEach((mesh) => enableToonShading(mesh));

        } else if (infos.parameters.get('use_light_toon_shader')) {
            for (let segment of robot.segments)
                segment.visual.meshes.forEach((mesh) => enableLightToonShading(mesh));

            for (let tool of robot.tools)
                tool.visual.meshes.forEach((mesh) => enableLightToonShading(mesh));
        }

        if (this.endEffectorManipulationEnabled && robot.controlsEnabled)
            robot._createTcpTarget();

        this.activateLayer(Layers.Base);

        return robot;
    }


    /**
     * Retrieve a robot instance by name.
     * @param {string} name - robot name
     * @returns {Robot|undefined} the Robot instance or undefined if not found
     */
    getRobot(name) {
        return this.robots[name];
    }

    /**
     * Create and add a manipulable target to the scene.
     * Targets are used as destination pose markers for end-effectors.
     *
     * @param {string} name - unique name of the target
     * @param {THREE.Vector3|Array<number>} position - position vector or array
     * @param {THREE.Quaternion|Array<number>} orientation - orientation quaternion or array
     * @param {number|string} [color=0x0000aa]
     * @param {Shapes} [shape=Shapes.Cube]
     * @param {(target:Object)=>void|null} [listener=null] - called when the target is moved
     * @param {Object|null} [parameters=null] - extra shape-dependent parameters
     * @returns {Target}
     */
    addTarget(name, position, orientation, color, shape=Shapes.Cube, listener=null, parameters=null) {
        const target = this.targets.create(name, position, orientation, color, shape, listener, parameters);

        target.layers.disableAll();
        target.layers.enable(this.activeLayer);

        this.scene.add(target);
        return target;
    }


    /**
     * Remove a target previously added.
     * @param {string} name
     * @returns {void}
     */
    removeTarget(name) {
        const target = this.targets.get(name);

        if (this.transformControls.getAttachedObject() == target)
            this.transformControls.detach();

        this.targets.destroy(name);
    }


    /**
     * Retrieve a target by name.
     * @param {string} name
     * @returns {Target|null}
     */
    getTarget(name) {
        return this.targets.get(name);
    }


    /**
     * Add a directional arrow helper to the scene.
     * @param {string} name
     * @param {THREE.Vector3|Array<number>} origin
     * @param {THREE.Vector3|Array<number>} direction - should be normalized
     * @param {number} [length=1]
     * @param {number|string} [color=0xffff00]
     * @param {boolean} [shading=false]
     * @param {number} [headLength=length*0.2]
     * @param {number} [headWidth=headLength*0.2]
     * @param {number} [radius=headWidth*0.1]
     * @returns {Arrow}
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


    /**
     * Remove an arrow helper by name.
     * @param {string} name
     * @returns {void}
     */
    removeArrow(name) {
        this.arrows.destroy(name);
    }


    /**
     * Get an arrow helper by name.
     * @param {string} name
     * @returns {Arrow|null}
     */
    getArrow(name) {
        return this.arrows.get(name);
    }


    /**
     * Add a small axes helper at a location.
     * @param {string} name
     * @param {THREE.Vector3|Array<number>|null} [position=null]
     * @param {THREE.Quaternion|Array<number>|null} [orientation=null]
     * @param {number} [length=0.1]
     * @returns {Axes}
     */
    addAxes(name, position=null, orientation=null, length=0.1) {
        const axes = new Axes(name, position, orientation, length);

        axes.layers.disableAll();
        axes.layers.enable(this.activeLayer);

        this.axes.add(axes);
        this.scene.add(axes);
        return axes;
    }


    /**
     * Remove axes helper by name.
     * @param {string} name
     * @returns {void}
     */
    removeAxes(name) {
        this.axes.destroy(name);
    }


    /**
     * Retrieve axes helper by name.
     * @param {string} name
     * @returns {Axes|null}
     */
    getAxes(name) {
        return this.axes.get(name);
    }


    /**
     * Add a polyline path to the scene.
     * @param {string} name
     * @param {Array<THREE.Vector3>|Array<Array<number>>} points
     * @param {number} [radius=0.01]
     * @param {number|string} [color=0xffff00]
     * @param {boolean} [shading=false]
     * @param {boolean} [transparent=false]
     * @param {number} [opacity=0.5]
     * @returns {Path}
     */
    addPath(name, points, radius=0.01, color=0xffff00, shading=false, transparent=false, opacity=0.5) {
        const path = new Path(name, points, radius, color, shading, transparent, opacity);

        path.layers.disableAll();
        path.layers.enable(this.activeLayer);

        this.paths.add(path);
        this.scene.add(path);
        return path;
    }


    /**
     * Remove a path by name.
     * @param {string} name
     * @returns {void}
     */
    removePath(name) {
        this.paths.destroy(name);
    }


    /**
     * Get a path by name.
     * @param {string} name
     * @returns {Path|null}
     */
    getPath(name) {
        return this.paths.get(name);
    }


    /**
     * Add a small point marker to the scene, optionally with a LaTeX label.
     * @param {string} name
     * @param {THREE.Vector3|Array<number>} position
     * @param {number} [radius=0.01]
     * @param {number|string} [color=0xffff00]
     * @param {string|null} [label=null]
     * @param {boolean} [shading=false]
     * @param {boolean} [transparent=false]
     * @param {number} [opacity=0.5]
     * @returns {Point}
     */
    addPoint(name, position, radius=0.01, color=0xffff00, label=null, shading=false, transparent=false, opacity=0.5) {
        const point = new Point(name, position, radius, color, label, shading, transparent, opacity);

        point.layers.disableAll();
        point.layers.enable(this.activeLayer);

        this.points.add(point);
        this.scene.add(point);
        return point;
    }


    /**
     * Remove a point marker by name.
     * @param {string} name
     * @returns {void}
     */
    removePoint(name) {
        this.points.destroy(name);
    }


    /**
     * Get a point marker by name.
     * @param {string} name
     * @returns {Point|null}
     */
    getPoint(name) {
        return this.points.get(name);
    }


    /**
     * Add a 3D Gaussian visualization to the scene.
     * @param {string} name
     * @param {THREE.Vector3|Array<number>} mu
     * @param {THREE.Matrix3|THREE.Matrix4|Array<number>} sigma - covariance
     * @param {number|string} [color=0xffff00]
     * @param {(gaussian:Object)=>void|null} [listener=null]
     * @returns {Gaussian}
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
        return this.physicsSimulator.getPhysicalBody(name);

    }


    /**
     * Translate the camera target by a delta vector and update controls.
     * @param {THREE.Vector3} delta
     * @returns {void}
     */
    translateCamera(delta) {
        this.cameraControl.target.add(delta);
        this.cameraControl.update();
    }


    /**
     * Enable the logmap overlay for a robot/target pair.
     * @param {string|Robot} robot - robot name or instance
     * @param {string|Target} target - target name or instance
     * @param {string} [position='left'] - 'left' or 'right', placement of the logmap
     * @param {Object|null} [size=null] - optional size {width,height}
     * @returns {void}
     */
    enableLogmap(robot, target, position='left', size=null) {
        if (typeof(robot) == "string")
            robot = this.getRobot(robot);

        if (typeof(target) == "string")
            target = this.getTarget(target);

        this.logmap = new Logmap(this.domElement, robot, target, size, position);
    }


    /**
     * Disable any active logmap overlay.
     * @returns {void}
     */
    disableLogmap() {
        this.logmap = null;
    }


    /**
     * Stop the viewer render loop and wait until it has fully stopped.
     * When using external_loop=true this is a no-op and returns immediately.
     * @returns {Promise<void>} resolves once the loop has stopped
     */
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
            ['shadows', true],
            ['show_joint_positions', false],
            ['show_axes', false],
            ['statistics', false],
            ['external_loop', false],
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


    _updatePhysics(elapsed) {
        if (this.physicsSimulator != null) {
            this.physicsSimulator.update(elapsed);
            this.physicsSimulator.synchronize();
        }
    }

    /**
     * Render a frame. This is the main loop entry point and is called repeatedly
     * when 'external_loop' is false.
     * @returns {void}
     */
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


        for (const name in this.robots) {
            const robot = this.robots[name];

            for (let i = 0; i < robot.tools.length; ++i) {
                const tool = robot.tools[i];

                if (robot.configuration.tools[i].type == 'gripper')
                    this.physicsSimulator.setControl(tool.ctrl, tool.actuators);
            }
        }


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
                    this._updatePhysics(this.renderingCallbackTime);

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
                this._updatePhysics(this.clock.elapsedTime);

                this.renderingCallback(delta, this.clock.elapsedTime);
            }
        }
        else
        {
            // Update the physics simulator
            this._updatePhysics(this.clock.elapsedTime);
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

            this.planarIkControls.reset();

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
    globalThis.SceneBuilder = SceneBuilder;
    globalThis.Robots = Robots;
    globalThis.Tools = Tools;
    globalThis.readFile = readFile;
    globalThis.writeFile = writeFile;

    globalThis.configs = {
        RobotConfiguration: RobotConfiguration,
        ToolConfiguration: ToolConfiguration,
        GripperConfiguration: GripperConfiguration,
        Panda: PandaConfiguration,
        FrankaHand: FrankaHandConfiguration,
        Robotiq2F85: Robotiq2F85Configuration,
        RobotiqHandE: RobotiqHandEConfiguration,
        G1: G1Configuration,
        G1FixedLegs: G1FixedLegsConfiguration,
        InspireRightRH56DFXConfiguration: InspireRightRH56DFXConfiguration,
        InspireLeftRH56DFXConfiguration: InspireLeftRH56DFXConfiguration,
        UnitreeRightHand: UnitreeRightHandConfiguration,
        UnitreeLeftHand: UnitreeLeftHandConfiguration,
        UnitreeRightDex31: UnitreeRightDex31Configuration,
        UnitreeLeftDex31: UnitreeLeftDex31Configuration,
        UR5: UR5Configuration,
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

export { FrankaHandConfiguration, G1Configuration, G1FixedLegsConfiguration, GripperConfiguration, InspireLeftRH56DFXConfiguration, InspireRightRH56DFXConfiguration, LayerRenderPass, Layers, OutlinePass, PandaConfiguration, Passes, RobotBuilder, RobotConfiguration, Robotiq2F85Configuration, RobotiqHandEConfiguration, Robots, SceneBuilder, Shapes, ToolConfiguration, Tools, UR5Configuration, UnitreeLeftDex31Configuration, UnitreeLeftHandConfiguration, UnitreeRightDex31Configuration, UnitreeRightHandConfiguration, Viewer3D, downloadFile, downloadFiles, downloadRobot, downloadScene, downloadTool, getURL, initPyScript, initViewer3D, matrixFromSigma, readFile, sigmaFromMatrix3, sigmaFromMatrix4, sigmaFromQuaternionAndScale, writeFile };
