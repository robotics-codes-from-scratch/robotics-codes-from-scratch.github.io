import { RawShaderMaterial, UniformsUtils, GLSL3, Vector3 } from 'three';
import { Pass, FullScreenQuad } from 'three/examples/jsm/postprocessing/Pass.js';

const SDFShader = {
    name: 'SDFShader',

    uniforms: {
        'tDiffuse': { value: null },
        'tDepth': { value: null },
        'target_position': { value: [0.3, 0.0, 0.3] },
        'sdf_disc_radius': { value: 0.08 },
        'sdf_box_size': { value: [0.24, 0.1, 0.12] },
        'sdf_box_offset': { value: [0.08, 0.06, -0.06] },
        'sdf_smoothing_ratio': { value: 0.05 },
        'camera_fovy': { value: null },
        'camera_position': { value: null },
        'camera_target': { value: null },
        'camera_up': { value: [0, 0, 1] },
        'camera_near': { value: 0.1 },
        'camera_far': { value: 100.0 },
        'iResolution': { value: [700, 700] },
        'iFrame': { value: 0 },
    },

    vertexShader: /* glsl */`
        precision highp float;

        uniform mat4 modelViewMatrix;
        uniform mat4 projectionMatrix;

        in vec3 position;
        in vec2 uv;

        out vec2 vUv;

        void main() {
            vUv = uv;
            gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
        }`,

    fragmentShader: /* glsl */`
        precision highp float;

        #include <packing>

        uniform sampler2D tDiffuse;
        uniform sampler2D tDepth;
        uniform vec3 target_position;
        uniform float sdf_disc_radius;
        uniform vec3 sdf_box_size;
        uniform vec3 sdf_box_offset;
        uniform float sdf_smoothing_ratio;
        uniform vec2 iResolution;
        uniform int iFrame;
        uniform float camera_fovy;
        uniform vec3 camera_position;
        uniform vec3 camera_target;
        uniform vec3 camera_up;
        uniform float camera_near;
        uniform float camera_far;

        in vec2 vUv;
        out vec4 outColor;

        // Constants
        const float PI = 3.14159265;


        float readDepth(sampler2D depthSampler, vec2 coord) {
            float depth = texture(depthSampler, coord).x;
            float viewZ = -perspectiveDepthToViewZ(depth, camera_near, camera_far);
            return viewZ;
        }


        // Rays
        struct ray_t {
            vec3 origin;
            vec3 direction;
        };

        vec3 evaluateRay(ray_t ray, float t)
        {
            return ray.origin + ray.direction * t;
        }


        // Camera
        struct camera_t {
            float fovy;
            vec3 position;
            vec3 target;
            vec3 up;
        };


        // Viewport
        struct viewport_t {
            vec2 size;
            vec3 u;
            vec3 v;
        };

        viewport_t createViewport(camera_t camera)
        {
            viewport_t viewport;

            float aspectRatio = float(iResolution.x) / float(iResolution.y);
            float h = tan(radians(camera.fovy) / 2.0);

            float focalLength = length(camera.position - camera.target);

            viewport.size.y = 2.0 * h * focalLength;
            viewport.size.x = viewport.size.y * aspectRatio;

            // viewport.size.y = 1.8176831792880865;
            // viewport.size.x = 1.8176831792880865;

            vec3 w = normalize(camera.position - camera.target);
            vec3 u = normalize(cross(camera.up, w));
            vec3 v = cross(w, u);

            // Calculate the vectors across the horizontal and down the vertical viewport edges
            viewport.u = viewport.size.x * u;
            viewport.v = viewport.size.y * v;

            return viewport;
        }


        // SDF for circle
        float sdf_circle(vec3 point, vec3 center, float radius)
        {
            return length(center - point) - radius;
        }

        // SDF for box
        float sdf_box(vec3 point, vec3 center, vec3 dimensions)
        {
            vec3 d = abs(center - point) - dimensions * 0.5;
            return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
        }

        float smooth_union(float d1, float d2, float k)
        {
            float h = max(k - abs(d1 - d2), 0.0);
            float d = min(d1, d2) - h * h * 0.25 / k;
            return d;
        }

        float computeDistance(vec3 position)
        {
            vec3 p1 = target_position;
            vec3 p2 = target_position + sdf_box_offset;
            float d1 = sdf_circle(position, p1, sdf_disc_radius);
            float d2 = sdf_box(position, p2, sdf_box_size);
            return smooth_union(d1, d2, sdf_smoothing_ratio);
        }


        // Represents the distance of a point to the nearest object, with the color of said object
        struct map_result_t
        {
            float distance;
            vec3 color;
        };

        // Returns the distance of a point to the nearest object, with the color of said object
        map_result_t map(vec3 position)
        {
            float dist = computeDistance(position);
            return map_result_t(dist, vec3(1.0, 0.2, 0.6));
        }


        // Represents the result of a raycast in the scene: did it hit an object, at which distance
        // (from the origin of the ray) and the color of the object
        struct raycast_result_t
        {
            bool hit;
            float distance;
            vec3 color;
        };

        // Performs raymarching through the scene, using the given ray
        raycast_result_t raycast(ray_t ray, float offset)
        {
            raycast_result_t result = raycast_result_t(false, 0.0, vec3(0.0, 0.0, 0.0));

            float tmin = 0.1;
            float tmax = 10.0;

            float t = tmin;
            for (int i = 0; (i < 70) && (t < tmax); ++i)
            {
                map_result_t h = map(evaluateRay(ray, t));
                if (abs(h.distance) < (0.0001 * t) + offset)
                {
                    result.hit = true;
                    result.distance = t;
                    result.color = h.color;
                    break;
                }

                t += h.distance;
            }

            return result;
        }


        // Compute the normal at a given position
        // See https://iquilezles.org/articles/normalsSDF
        vec3 computeNormal(vec3 position)
        {
            #define ZERO (min(int(iFrame), 0)) // non-constant zero, to prevent the compiler to inline the map() function 4 times

            vec3 n = vec3(0.0);
            for (int i = ZERO; i < 4; ++i)
            {
                vec3 e = 0.5773 * (2.0 * vec3((((i+3)>>1)&1), ((i>>1)&1), (i&1)) - 1.0);
                n += e * map(position + 0.0005 * e).distance;
            }

            return normalize(n);
        }


        // Compute the fragment color for a given ray
        vec3 render(ray_t ray, camera_t camera, out bool hit, out float depth)
        { 
            // Background color
            vec3 color = vec3(0, 0, 0);

            // Raymarching into the scene
            raycast_result_t result = raycast(ray, 0.0);
            if (result.hit)
            {
                vec3 position = evaluateRay(ray, result.distance);

                // Determine if an object is closer than the SDF (in which case this is not a hit)
                depth = readDepth(tDepth, vUv);

                vec3 ray_dir = position - camera.position;
                vec3 camera_dir = camera.target - camera.position;

                float ray_depth = dot(ray_dir, camera_dir) / length(camera_dir);

                if (depth < ray_depth)
                    return color;

                // Compute the color of the pixel (with one point light and one directional light, hardcoded)
                vec3 normal = computeNormal(position);

                color = vec3(0.0);

                // Ambient light
                {
                    color += 0.2 * result.color;
                }

                // Point light
                {
                    vec3 lightDirection = normalize(vec3(3.0, 4.0, 3.0));
                    vec3 halfDirection = normalize(lightDirection - ray.direction);

                    float diffuse = clamp(dot(normal, lightDirection), 0.0, 1.0);
                    float specular = pow(clamp(dot(normal, halfDirection), 0.0, 1.0), 8.0);
                    specular *= diffuse;
                    specular *= 0.04 + 0.96 * pow(clamp(1.0 -dot(halfDirection, lightDirection), 0.0, 1.0), 1.0);
                    color += result.color * 0.5 * diffuse * vec3(1.0, 1.0, 1.0);
                    color += specular * vec3(1.0, 1.0, 1.0);
                }

                // Directional light
                {
                    vec3 lightDirection = normalize(vec3(0.0, 0.0, 1.0));
                    float diffuse = clamp(dot(normal, lightDirection), 0.0, 1.0);
                    color += result.color * 2.2 * diffuse * vec3(0.5, 0.5, 0.5);
                }

                hit = true;
                depth = result.distance;
            }

            return vec3(clamp(color, 0.0, 1.0));
        }


        void main()
        {
            // Setup the camera
            camera_t camera = camera_t(
                camera_fovy,
                camera_position,
                camera_target,
                camera_up
            );

            // Setup the viewport
            viewport_t viewport = createViewport(camera);

            // Calculate the horizontal and vertical delta vectors from pixel to pixel
            vec3 dU = viewport.u / float(iResolution.x);
            vec3 dV = viewport.v / float(iResolution.y);

            // Calculate the location of the lower left pixel, which is at vUv = (0, 0)
            vec3 viewportLowerLeft = camera.target - viewport.u / 2.0 - viewport.v / 2.0;
            vec3 pixel00 = viewportLowerLeft + 0.5 * (dU + dV);

            // Create the ray
            vec3 pixelCenter = pixel00 + (vUv.x * viewport.u) + (vUv.y * viewport.v);
            ray_t ray = ray_t(camera.position, normalize(pixelCenter - camera.position));

            // Rendering
            bool hit;
            float depth;
            vec3 color = render(ray, camera, hit, depth);

            if (hit)
                outColor = vec4(color, 1.0);
            else
                outColor = texture(tDiffuse, vUv);

            gl_FragDepth = hit ? depth : readDepth(tDepth, vUv);
        }`
};


class SDFPass extends Pass {

    constructor(viewer3d) {
        super();

        this.camera = viewer3d.camera;
        this.cameraControl = viewer3d.cameraControl;
        this.domElement = viewer3d.domElement;

        const shader = SDFShader;

        this.uniforms = UniformsUtils.clone(shader.uniforms);

        this.counter = 0;

        this.material = new RawShaderMaterial({
            name: shader.name,
            uniforms: this.uniforms,
            vertexShader: shader.vertexShader,
            fragmentShader: shader.fragmentShader,
            glslVersion: GLSL3,
        });

        this.fsQuad = new FullScreenQuad(this.material);
    }

    setSDF(target_position, sdf_disc_radius, sdf_box_size, sdf_box_offset, sdf_smoothing_ratio) {
        this.uniforms[ 'target_position' ].value = target_position;
        this.uniforms[ 'sdf_disc_radius' ].value = sdf_disc_radius;
        this.uniforms[ 'sdf_box_size' ].value = sdf_box_size;
        this.uniforms[ 'sdf_box_offset' ].value = sdf_box_offset;
        this.uniforms[ 'sdf_smoothing_ratio' ].value = sdf_smoothing_ratio;
    }

    render(renderer, writeBuffer, readBuffer/*, deltaTime, maskActive */) {
        const cameraPosition = new three.Vector3();
        this.camera.getWorldPosition(cameraPosition);

        this.uniforms[ 'tDiffuse' ].value = readBuffer.texture;
        this.uniforms[ 'tDepth' ].value = readBuffer.depthTexture;
        this.uniforms[ 'camera_fovy' ].value = this.camera.fov;
        this.uniforms[ 'camera_position' ].value = cameraPosition;
        this.uniforms[ 'camera_target' ].value = this.cameraControl.target.clone();
        this.uniforms[ 'camera_near' ].value = this.camera.near;
        this.uniforms[ 'camera_far' ].value = this.camera.far;
        this.uniforms[ 'iResolution' ].value = [this.domElement.clientWidth, this.domElement.clientHeight];
        this.uniforms[ 'iFrame' ].value = this.counter;

        // if (this.counter % 1000 == 0)
        // if (this.counter == 0)
        //     this.debug();

        renderer.setRenderTarget(writeBuffer);
        this.fsQuad.render(renderer);

        ++this.counter;
    }

    dispose() {
        this.material.dispose();
        this.fsQuad.dispose();
    }

    // debug() {
    //     console.log('-----------------');
    //
    //     const iResolution_x = 700;
    //     const iResolution_y = 700;
    //
    //     const cameraPosition = new three.Vector3();
    //     this.camera.getWorldPosition(cameraPosition);
    //
    //     const camera = {
    //         fovy: this.camera.fov,
    //         position: cameraPosition,
    //         target: new three.Vector3(0.0, 0.0, 0.4),
    //         up: new three.Vector3(0.0, 0.0, 1.0),
    //     };
    //
    //     const viewport = this.createViewport(camera);
    //
    //     console.log('camera = ', camera);
    //     console.log('viewport = ', viewport);
    //
    //     // Calculate the horizontal and vertical delta vectors from pixel to pixel
    //     const dU = viewport.u.clone();
    //     dU.divideScalar(iResolution_x);
    //
    //     const dV = viewport.v.clone();
    //     dV.divideScalar(iResolution_y);
    //
    //     // Calculate the location of the lower left pixel, which is at vUv = (0, 0)
    //     const viewportLowerLeft = camera.target.clone();
    //
    //     const diffU = viewport.u.clone();
    //     diffU.divideScalar(2.0);
    //
    //     const diffV = viewport.v.clone();
    //     diffV.divideScalar(2.0);
    //
    //     viewportLowerLeft.sub(diffU);
    //     viewportLowerLeft.sub(diffV);
    //
    //     const pixel00 = dU.clone();
    //     pixel00.add(dV);
    //     pixel00.multiplyScalar(0.5);
    //     pixel00.add(viewportLowerLeft);
    //
    //     console.log('viewportLowerLeft = ', viewportLowerLeft);
    //     console.log('pixel00 = ', pixel00);
    //
    //     const uvs = [
    //         new three.Vector2(0.0, 0.0),
    //         new three.Vector2(1.0, 0.0),
    //         new three.Vector2(0.0, 1.0),
    //         new three.Vector2(1.0, 1.0),
    //         new three.Vector2(0.5, 0.5),
    //     ];
    //
    //     for (let i = 0; i < uvs.length; ++i)
    //     {
    //         console.log('UV #', i, ' = ', uvs[i]);
    //
    //         // Create the ray
    //         // const dU2 = dU.clone();
    //         const dU2 = viewport.u.clone();
    //         dU2.multiplyScalar(uvs[i].x);
    //
    //         // const dV2 = dV.clone();
    //         const dV2 = viewport.v.clone();
    //         dV2.multiplyScalar(uvs[i].y);
    //
    //         const pixelCenter = pixel00.clone();
    //         pixelCenter.add(dU2);
    //         pixelCenter.add(dV2);
    //
    //         console.log('    pixelCenter = ', pixelCenter);
    //
    //         const ray = {
    //             origin: camera.position.clone(),
    //             direction: null
    //         };
    //
    //         ray.direction = pixelCenter.clone();
    //         ray.direction.sub(camera.position);
    //         ray.direction.normalize();
    //
    //         console.log('    ray = ', ray);
    //
    //         const raycast_result = this.raycast(ray);
    //
    //         console.log('    raycast_result = ', raycast_result);
    //     }
    // }
    //
    // createViewport(camera) {
    //     const viewport = {
    //         size: new three.Vector2(),
    //         u: new three.Vector3(),
    //         v: new three.Vector3(),
    //     };
    //
    //     const iResolution_x = 700;
    //     const iResolution_y = 700;
    //
    //     const aspectRatio = iResolution_x / iResolution_y;
    //     const h = Math.tan(camera.fovy * Math.PI / 180.0 / 2.0);
    //
    //     const focalLength = camera.position.distanceTo(camera.target);
    //
    //     viewport.size.y = 2.0 * h * focalLength;
    //     viewport.size.x = viewport.size.y * aspectRatio;
    //
    //     const w = new three.Vector3();
    //     w.subVectors(camera.position, camera.target);
    //     w.normalize();
    //
    //     const u = new three.Vector3();
    //     u.crossVectors(camera.up, w);
    //     u.normalize();
    //
    //     const v = new three.Vector3();
    //     v.crossVectors(w, u);
    //
    //     // Calculate the vectors across the horizontal and down the vertical viewport edges
    //     viewport.u.copy(u);
    //     viewport.u.multiplyScalar(viewport.size.x);
    //
    //     viewport.v.copy(v);
    //     viewport.v.multiplyScalar(viewport.size.y);
    //
    //     return viewport;
    // }
    //
    // evaluateRay(ray, t) {
    //     const res = ray.direction.clone();
    //     res.multiplyScalar(t);
    //     res.add(ray.origin);
    //     return res;
    // }
    //
    //
    // // Performs raymarching through the scene, using the given ray
    // raycast(ray) {
    //     const raycast_result = {
    //         hit: false,
    //         distance: 0.0,
    //         color: new three.Vector3(0.0, 0.0, 0.0)
    //     };
    //
    //     const tmin = 0.1;
    //     const tmax = 4.0;
    //
    //     let t = tmin;
    //     for (let i = 0; (i < 70) && (t < tmax); ++i)
    //     {
    //         const h = this.map(this.evaluateRay(ray, t));
    //         console.log('    t = ', t);
    //         console.log('    h = ', h);
    //         if (Math.abs(h.distance) < (0.0001 * t))
    //         {
    //             raycast_result.hit = true;
    //             raycast_result.distance = t;
    //             raycast_result.color = h.color;
    //             break;
    //         }
    //
    //         t += h.distance;
    //     }
    //
    //     return raycast_result;
    // }
    //
    // sdf_circle(point, center, radius) {
    //     const diff = center.clone();
    //     diff.sub(point);
    //     return diff.length() - radius;
    // }
    //
    // computeDistance(position) {
    //     const p1 = new three.Vector3(0.0, 0.0, 0.4);
    //     return this.sdf_circle(position, p1, 0.04);
    // }
    //
    // // Returns the distance of a point to the nearest object, with the color of said object
    // map(position) {
    //     const dist = this.computeDistance(position);
    //     return {
    //         distance: dist,
    //         color: new three.Vector3(0.5, 0.0, 0.0)
    //     };
    // }
};

globalThis.SDFPass = SDFPass;
