/*
 * SPDX-FileCopyrightText: Copyright © 2023 Idiap Research Institute <contact@idiap.ch>
 *
 * SPDX-FileContributor: Philip Abbet <philip.abbet@idiap.ch>
 *
 * SPDX-License-Identifier: MIT
 *
 */

import load_mujoco from "mujoco";
import * as THREE from 'three';

import { getURL } from './utils.js';
import Robot from './robots/robot.js';


// Load the MuJoCo Module
const mujoco = await load_mujoco();



/* Download some files and store them in Mujoco's filesystem

Parameters:
    dstFolder (string): The destination folder in Mujoco's filesystem
    srcFolderUrl (string): The URL of the folder in which all the files are located
    filenames ([string]): List of the filenames
*/
export async function downloadFiles(dstFolder, srcFolderUrl, filenames) {
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

        if (filename.endsWith(".png") || filename.endsWith(".stl") || filename.endsWith(".skn")) {
            mujoco.FS.writeFile(dstFolder + '/' + filename, new Uint8Array(await data.arrayBuffer()));
        } else {
            mujoco.FS.writeFile(dstFolder + '/' + filename, await data.text());
        }
    }
}



/* Download a scene file

The scenes are stored in Mujoco's filesystem at '/scenes'
*/
export async function downloadScene(url, destFolder='/scenes') {
    const offset = url.lastIndexOf('/');
    await downloadFiles(destFolder, url.substring(0, offset), [ url.substring(offset + 1) ]);
}



/* Download all the files needed to simulate and display the Franka Emika Panda robot

The files are stored in Mujoco's filesystem at '/scenes/franka_emika_panda'
*/
export async function downloadPandaRobot() {
    const dstFolder = '/scenes/franka_emika_panda';
    const srcURL = getURL('models/franka_emika_panda/');

    await downloadFiles(
        dstFolder,
        srcURL,
        ['panda.xml']
    );

    await downloadFiles(
        dstFolder + '/assets',
        srcURL + 'assets/',
        [
            'link0_c.obj',
            'link0_c.obj',
            'link1_c.obj',
            'link2_c.obj',
            'link3_c.obj',
            'link4_c.obj',
            'link5_collision_0.obj',
            'link5_collision_1.obj',
            'link5_collision_2.obj',
            'link6_c.obj',
            'link7_c.obj',
            'hand_c.obj',
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



export function loadScene(filename) {
    // Retrieve some infos from the XML file (not exported by the MuJoCo API)
    const xmlDoc = loadXmlFile(filename);
    if (xmlDoc == null)
        return null;

    const freeCameraSettings = getFreeCameraSettings(xmlDoc);
    const statistics = getStatistics(xmlDoc);
    const fogSettings = getFogSettings(xmlDoc);
    const headlightSettings = getHeadlightSettings(xmlDoc);

    // Load in the state from XML
    let model = new mujoco.Model(filename);
    let state = new mujoco.State(model);
    let simulation = new mujoco.Simulation(model, state);

    return new PhysicsSimulator(
        model, state, simulation, freeCameraSettings, statistics, fogSettings, headlightSettings
    );
}



class PhysicsSimulator {

    constructor(model, state, simulation, freeCameraSettings, statistics, fogSettings, headlightSettings) {
        this.model = model;
        this.state = state;
        this.simulation = simulation;
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
        this._processLights();
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


    update(time) {
        if (!this.paused) {
            let timestep = this.model.getOptions().timestep;

            if (time - this.time > 0.035)
                this.time = time;

            while (this.time < time) {
                this.simulation.step();
                this.time += timestep;
            }
        } else {
            this.simulation.forward();
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

        // // Update site transforms
        // for (let s = 0; s < this.model.nsite; ++s) {
        //     const site = this.sites[s];
        //     const body_id = this.model.site_bodyid[s];
        //
        //     if (body_id > 0) {
        //         const body = this.bodies[body_id];
        //
        //         this._getPosition(this.simulation.site_xpos, s, pos);
        //
        //         body.worldToLocal(pos);
        //         site.position.copy(pos);
        //
        //     } else {
        //         this._getPosition(this.simulation.site_xpos, s, site.position);
        //     }
        //     // this._getPosition(this.simulation.site_xpos, s, site.position);
        //
        //     console.log(s, site.position);
        //
        //     // this._getQuaternion(this.simulation.geom_xmat, s, site.quaternion);
        // }
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


    createRobot(name, configuration) {
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

        // Create the robot
        const robot = new Robot(name, configuration, this);

        // Search the root body of the robot and the tool (if any)
        let rootBody = null;
        let toolBody = null;
        for (let b = 0; b < this.model.nbody; ++b) {
            const name = this.names[this.model.name_bodyadr[b]];

            if ((rootBody == null) && (name == configuration.robotRoot))
                rootBody = b;

            if ((toolBody == null) && (name == configuration.toolRoot))
                toolBody = b;

            if (rootBody != null) {
                if ((configuration.toolRoot == null) || (toolBody != null))
                    break;
            }
        }

        if (rootBody == null) {
            console.error("Failed to create the robot: link '" + configuration.robotRoot + "' not found");
            return null;
        }

        // Retrieve all the bodies of arm of the robot
        robot.arm.links = [rootBody];
        _getChildBodies(rootBody, robot.arm.links);

        // Retrieve all the bodies of the tool of the robot
        if (toolBody != null) {
            robot.tool.links = [toolBody];
            _getChildBodies(toolBody, robot.tool.links);
        }

        // Retrieve all the joints of the robot
        for (let j = 0; j < this.model.njnt; ++j) {
            const body = this.model.jnt_bodyid[j];

            if (robot.arm.links.indexOf(body) >= 0) {
                robot.arm.joints.push(j);
                this.freeJoints.splice(this.freeJoints.indexOf(j), 1)

            } else if (robot.tool.links.indexOf(body) >= 0) {
                robot.tool.joints.push(j);
                this.freeJoints.splice(this.freeJoints.indexOf(j), 1)
            }
        }

        // Retrieve all the actuators of the robot
        for (let a = 0; a < this.model.nu; ++a) {
            const type = this.model.actuator_trntype[a];
            const id = this.model.actuator_trnid[a * 2];

            if ((type == mujoco.mjtTrn.mjTRN_JOINT.value) ||
                (type == mujoco.mjtTrn.mjTRN_JOINT.mjTRN_JOINTINPARENT)) {

                if (robot.arm.joints.indexOf(id) >= 0)
                    robot.arm.actuators.push(a);
                else if (robot.tool.joints.indexOf(id) >= 0)
                    robot.tool.actuators.push(a);

            } else if (type == mujoco.mjtTrn.mjTRN_TENDON.value) {
                const adr = this.model.tendon_adr[id];
                const nb = this.model.tendon_num[id];

                for (let w = adr; w < adr + nb; ++w) {
                    if (this.model.wrap_type[w] == mujoco.mjtWrap.mjWRAP_JOINT.value) {
                        const jointId = this.model.wrap_objid[w];

                        if (robot.arm.joints.indexOf(jointId) >= 0)
                            robot.arm.actuators.push(a);
                        else if (robot.tool.joints.indexOf(jointId) >= 0)
                            robot.tool.actuators.push(a);

                        break;
                    }
                }
            }
        }

        // Retrieve the TCP of the robot (if necessary)
        if (configuration.tcpSite != null) {
            for (let s = 0; s < this.model.nsite; ++s) {
                const name = this.names[this.model.name_siteadr[s]];

                if (name == configuration.tcpSite) {
                    robot.tcp = this.sites[s];
                    break;
                }
            }
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
            if (type == mujoco.mjtGeom.mjGEOM_PLANE.value) {
                // Special handling for plane later
            } else if (type == mujoco.mjtGeom.mjGEOM_HFIELD.value) {
                // TODO: Implement this
            } else if (type == mujoco.mjtGeom.mjGEOM_SPHERE.value) {
                geometry = new THREE.SphereGeometry(size[0]);
            } else if (type == mujoco.mjtGeom.mjGEOM_CAPSULE.value) {
                geometry = new THREE.CapsuleGeometry(size[0], size[1] * 2.0, 20, 20);
            } else if (type == mujoco.mjtGeom.mjGEOM_ELLIPSOID.value) {
                geometry = new THREE.SphereGeometry(1); // Stretch this below
            } else if (type == mujoco.mjtGeom.mjGEOM_CYLINDER.value) {
                geometry = new THREE.CylinderGeometry(size[0], size[0], size[1] * 2.0);
            } else if (type == mujoco.mjtGeom.mjGEOM_BOX.value) {
                geometry = new THREE.BoxGeometry(size[0] * 2.0, size[2] * 2.0, size[1] * 2.0);
            } else if (type == mujoco.mjtGeom.mjGEOM_MESH.value) {
                let meshID = this.model.geom_dataid[g];

                if (!(meshID in this.meshes)) {
                    geometry = new THREE.BufferGeometry();

                    // Positions
                    let vertex_buffer = this.model.mesh_vert.subarray(
                        this.model.mesh_vertadr[meshID] * 3,
                        (this.model.mesh_vertadr[meshID] + this.model.mesh_vertnum[meshID]) * 3
                    );

                    for (let v = 0; v < vertex_buffer.length; v += 3) {
                        let temp = vertex_buffer[v + 1];
                        vertex_buffer[v + 1] = vertex_buffer[v + 2];
                        vertex_buffer[v + 2] = -temp;
                    }

                    // Normals
                    let normal_buffer = this.model.mesh_normal.subarray(
                        this.model.mesh_vertadr[meshID] * 3,
                        (this.model.mesh_vertadr[meshID] + this.model.mesh_vertnum[meshID]) * 3
                    );

                    for (let v = 0; v < normal_buffer.length; v += 3) {
                        let temp = normal_buffer[v + 1];
                        normal_buffer[v + 1] = normal_buffer[v + 2];
                        normal_buffer[v + 2] = -temp;
                    }

                    // UVs
                    let uv_buffer = this.model.mesh_texcoord.subarray(
                        this.model.mesh_texcoordadr[meshID] * 2,
                        (this.model.mesh_texcoordadr[meshID] + this.model.mesh_vertnum[meshID]) * 2
                    );

                    // Indices
                    let triangle_buffer = this.model.mesh_face.subarray(
                        this.model.mesh_faceadr[meshID] * 3,
                        (this.model.mesh_faceadr[meshID] + this.model.mesh_facenum[meshID]) * 3
                    );

                    geometry.setAttribute("position", new THREE.BufferAttribute(vertex_buffer, 3));
                    geometry.setAttribute("normal", new THREE.BufferAttribute(normal_buffer, 3));
                    geometry.setAttribute("uv", new THREE.BufferAttribute(uv_buffer, 2));
                    geometry.setIndex(Array.from(triangle_buffer));

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
                let texId = this.model.mat_texid[matId];
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
                        texture.repeat.y = texrepeat_v
                    }

                    material = new THREE.MeshPhysicalMaterial({
                        color: new THREE.Color(color[0], color[1], color[2]),
                        transparent: color[3] < 1.0,
                        opacity: color[3],
                        specularIntensity: this.model.mat_specular[matId] * 0.5,
                        reflectivity: this.model.mat_reflectance[matId],
                        roughness: 1.0 - this.model.mat_shininess[matId],
                        metalness: 0.1,
                        map: texture
                    });

                } else if (material.color.r != color[0] ||
                           material.color.g != color[1] ||
                           material.color.b != color[2] ||
                           material.opacity != color[3]) {

                    material = new THREE.MeshPhysicalMaterial({
                        color: new THREE.Color(color[0], color[1], color[2]),
                        transparent: color[3] < 1.0,
                        opacity: color[3],
                        specularIntensity: this.model.mat_specular[matId] * 0.5,
                        reflectivity: this.model.mat_reflectance[matId],
                        roughness: 1.0 - this.model.mat_shininess[matId],
                        metalness: 0.1,
                    });
                }

            } else if (material.color.r != color[0] ||
                       material.color.g != color[1] ||
                       material.color.b != color[2] ||
                       material.opacity != color[3]) {

                material = new THREE.MeshPhysicalMaterial({
                    color: new THREE.Color(color[0], color[1], color[2]),
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
                mesh.rotateX(-Math.PI / 2);
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
                mesh.rotateX(-Math.PI / 2);

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
                mesh.scale.set(size[0], size[2], size[1])
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


    _processLights() {
        const sim = this;

        function _createOrUpdateAmbientLight(color) {
            if (sim.ambientLight == null) {
                sim.ambientLight = new THREE.AmbientLight(sim.headlightSettings.ambient);
                sim.ambientLight.layers.enableAll();
                sim.root.add(sim.ambientLight);
            } else {
                sim.ambientLight.color += color;
            }
        }

        if (this.headlightSettings.active) {
            if ((this.headlightSettings.ambient.r > 0.0) || (this.headlightSettings.ambient.g > 0.0) ||
                (this.headlightSettings.ambient.b > 0.0)) {
                _createOrUpdateAmbientLight(this.headlightSettings.ambient);
            }

            if ((this.headlightSettings.diffuse.r > 0.0) || (this.headlightSettings.diffuse.g > 0.0) ||
                (this.headlightSettings.diffuse.b > 0.0)) {
                this.headlight = new THREE.DirectionalLight(this.headlightSettings.diffuse);
                this.headlight.layers.enableAll();
                this.root.add(this.headlight);
            }
        }

        const dir = new THREE.Vector3();

        for (let l = 0; l < this.model.nlight; ++l) {
            let light = null;

            if (this.model.light_directional[l])
                light = new THREE.DirectionalLight();
            else
                light = new THREE.SpotLight();

            light.quaternion.set(0, 0, 0, 1);

            this._getPosition(this.model.light_pos, l, light.position);

            this._getPosition(this.model.light_dir, l, dir);
            dir.add(light.position);

            light.target.position.copy(dir);

            light.color.r = this.model.light_diffuse[l * 3];
            light.color.g = this.model.light_diffuse[l * 3 + 1];
            light.color.b = this.model.light_diffuse[l * 3 + 2];

            if (!this.model.light_directional[l]) {
                light.distance = this.model.light_attenuation[l * 3 + 1];
                light.penumbra = 0.5;
                light.angle = this.model.light_cutoff[l] * Math.PI / 180.0;
                light.castShadow = this.model.light_castshadow[l];

                light.shadow.camera.near = 0.1;
                light.shadow.camera.far = 50;
                // light.shadow.bias = 0.0001;
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
                    new THREE.Color(
                        this.model.light_ambient[l * 3],
                        this.model.light_ambient[l * 3 + 1],
                        this.model.light_ambient[l * 3 + 2])
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
        let rgbArray = this.model.tex_rgb;
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

                texture.colorSpace = THREE.LinearSRGBColorSpace;
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

            this.textures[texId] = texture;
            return texture;
        }
    }


    /** Access the vector at index, swizzle for three.js, and apply to the target THREE.Vector3
     * @param {Float32Array|Float64Array} buffer
     * @param {number} index
     * @param {THREE.Vector3} target */
    _getPosition(buffer, index, target, swizzle = true) {
        if (swizzle) {
            return target.set(
                buffer[(index * 3) + 0],
                buffer[(index * 3) + 2],
                -buffer[(index * 3) + 1]);
        } else {
            return target.set(
                buffer[(index * 3) + 0],
                buffer[(index * 3) + 1],
                buffer[(index * 3) + 2]);
        }
    }


    /** Access the quaternion at index, swizzle for three.js, and apply to the target THREE.Quaternion
     * @param {Float32Array|Float64Array} buffer
     * @param {number} index
     * @param {THREE.Quaternion} target */
    _getQuaternion(buffer, index, target, swizzle = true) {
        if (swizzle) {
            return target.set(
                -buffer[(index * 4) + 1],
                -buffer[(index * 4) + 3],
                buffer[(index * 4) + 2],
                -buffer[(index * 4) + 0]);
        } else {
            return target.set(
                buffer[(index * 4) + 0],
                buffer[(index * 4) + 1],
                buffer[(index * 4) + 2],
                buffer[(index * 4) + 3]);
        }
    }


    _getMatrix(buffer, index, target, swizzle = true) {
        if (swizzle) {
            return target.set(
                buffer[(index * 9) + 0],
                buffer[(index * 9) + 2],
                -buffer[(index * 9) + 1],
                buffer[(index * 9) + 6],
                buffer[(index * 9) + 8],
                -buffer[(index * 9) + 7],
                -buffer[(index * 9) + 3],
                -buffer[(index * 9) + 5],
                buffer[(index * 9) + 4]
            );
        } else {
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
        const tmp = this.statistics.center.z;
        this.statistics.center.z = -this.statistics.center.y;
        this.statistics.center.y = tmp;

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
        statistics.center = new THREE.Vector3(Number(v[0]), Number(v[2]), -Number(v[1]));
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
            settings.fog.setRGB(Number(v[0]), Number(v[1]), Number(v[2]));
            settings.fogEnabled = true;
        }

        value = xmlRgba.getAttribute("haze");
        if (value != null) {
            const v = value.split(" ");
            settings.haze.setRGB(Number(v[0]), Number(v[1]), Number(v[2]), Number(v[3]));
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


function getHeadlightSettings(xmlDoc) {
    const settings = {
        ambient: new THREE.Color(0.1, 0.1, 0.1),
        diffuse: new THREE.Color(0.4, 0.4, 0.4),
        active: true,
    };

    const xmlVisual = getFirstElementByTagName(xmlDoc, "visual");
    if (xmlVisual == null)
        return settings;

    const xmlHeadlight = getFirstElementByTagName(xmlVisual, "headlight");
    if (xmlHeadlight != null) {
        let value = xmlHeadlight.getAttribute("ambient");
        if (value != null) {
            const v = value.split(" ");
            settings.ambient.setRGB(Number(v[0]), Number(v[1]), Number(v[2]));
        }

        value = xmlHeadlight.getAttribute("diffuse");
        if (value != null) {
            const v = value.split(" ");
            settings.diffuse.setRGB(Number(v[0]), Number(v[1]), Number(v[2]), Number(v[3]));
        }

        value = xmlHeadlight.getAttribute("active");
        if (value != null)
            settings.active = (value == "1");
    }

    return settings;
}


function getFirstElementByTagName(xmlParent, name) {
    const xmlElements = xmlParent.getElementsByTagName(name);
    if (xmlElements.length > 0)
        return xmlElements[0];

    return null;
}
