import moderngl
import numpy as np
import pygltflib
import cv2
from pyrr import Matrix44


VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in float in_longitudinal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 v_normal;
out vec2 v_uv;
out float v_longitudinal;

void main() {
    v_normal = mat3(transpose(inverse(model))) * in_normal;
    v_uv = in_uv;
    v_longitudinal = in_longitudinal;
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec3 v_normal;
in vec2 v_uv;
in float v_longitudinal;

uniform sampler2D albedo_map;
uniform sampler2D metallic_roughness_map;
uniform sampler2D emissive_map;

uniform float emissive_intensity;
uniform int use_emissive;

out vec4 frag_color;

void main() {
    vec3 albedo = texture(albedo_map, v_uv).rgb;
    vec3 metallic_roughness = texture(metallic_roughness_map, v_uv).rgb;
    float metallic = metallic_roughness.b;
    float roughness = clamp(metallic_roughness.g, 0.04, 1.0);

    vec3 N = normalize(v_normal);
    vec3 L = normalize(vec3(-0.35, 0.45, 1.0));
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);

    float ambient = 0.42;
    float diffuse = 0.58 * NdotL;
    float specular_power = mix(64.0, 4.0, roughness);
    float specular = pow(NdotH, specular_power) * mix(0.08, 0.35, metallic);

    vec3 emissive = vec3(0.0);
    if (use_emissive != 0) {
        float texture_mask = texture(emissive_map, v_uv).r;
        float tip_mask = smoothstep(0.90, 0.985, v_longitudinal);
        float ember_mask = max(texture_mask, tip_mask);
        emissive = ember_mask * emissive_intensity * vec3(1.0, 0.28, 0.03) * 1.8;
    }

    vec3 color = albedo * (ambient + diffuse) + vec3(specular) + emissive;
    frag_color = vec4(color, 1.0);
}
"""


class Cigarette3DRenderer:
    def __init__(self, model_path, config=None):
        self.config = config or {}
        self.model_scale = self.config.get('model_scale', 1.0)
        self.model_offset = np.array([
            self.config.get('model_offset_x', 0.0),
            self.config.get('model_offset_y', 0.0),
            self.config.get('model_offset_z', 0.0)
        ], dtype=np.float32)
        self.rotation_offset = np.array([
            self.config.get('model_rotation_offset_x', 0.0),
            self.config.get('model_rotation_offset_y', 0.0),
            self.config.get('model_rotation_offset_z', 0.0)
        ], dtype=np.float32)
        
        self.ctx = None
        self.program = None
        self.vao = None
        self.vbo = None
        self.ibo = None
        self.textures = {}
        self.fbo = None
        self.fbo_texture = None
        self.fbo_depth = None

        self._vertex_count = 0
        self._index_count = 0
        self._model_extent = np.ones(3, dtype=np.float32)

        try:
            self._init_gl()
            gltf = pygltflib.GLTF2().load(str(model_path))
            buffer_data = gltf.binary_blob()
            self._load_model(gltf, buffer_data)
            self._load_textures(gltf, buffer_data)
            self._setup_framebuffer()
        except Exception:
            self.close()
            raise

        # Projection is updated for the current video frame size.
        self.view_matrix = Matrix44.identity()
        self.projection_matrix = Matrix44.identity()

        # Glow state
        self.emissive_intensity = 0.0
        self.target_emissive_intensity = 0.0
        self.fade_in_speed = self.config.get('glow_fade_in', 0.15)
        self.fade_out_speed = self.config.get('glow_fade_out', 0.08)

    def _init_gl(self):
        """Initialize ModernGL context for offscreen rendering."""
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST)
        
        # Compile shaders
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )

    @staticmethod
    def _accessor_data(gltf, buffer_data, accessor_index, components, dtype):
        """Read a tightly packed GLB accessor with safe zero offsets."""
        accessor = gltf.accessors[accessor_index]
        view = gltf.bufferViews[accessor.bufferView]
        offset = (view.byteOffset or 0) + (accessor.byteOffset or 0)
        item_size = np.dtype(dtype).itemsize * components
        stride = view.byteStride or item_size

        if stride == item_size:
            length = accessor.count * item_size
            return np.frombuffer(buffer_data[offset:offset + length], dtype=dtype).reshape(-1, components)

        values = np.empty((accessor.count, components), dtype=dtype)
        for index in range(accessor.count):
            start = offset + index * stride
            values[index] = np.frombuffer(
                buffer_data[start:start + item_size], dtype=dtype, count=components
            )
        return values

    def _load_model(self, gltf, buffer_data):
        """Load GLB model and extract mesh data."""
        mesh = gltf.meshes[0]
        prim = mesh.primitives[0]
        idx_acc = gltf.accessors[prim.indices]

        pos_data = self._accessor_data(
            gltf, buffer_data, prim.attributes.POSITION, 3, np.dtype('<f4')
        )
        norm_data = self._accessor_data(
            gltf, buffer_data, prim.attributes.NORMAL, 3, np.dtype('<f4')
        )
        uv_data = self._accessor_data(
            gltf, buffer_data, prim.attributes.TEXCOORD_0, 2, np.dtype('<f4')
        )
        index_types = {
            5121: np.dtype('u1'),
            5123: np.dtype('<u2'),
            5125: np.dtype('<u4'),
        }
        if idx_acc.componentType not in index_types:
            raise ValueError(f"Unsupported GLB index component type: {idx_acc.componentType}")
        idx_data = self._accessor_data(
            gltf, buffer_data, prim.indices, 1, index_types[idx_acc.componentType]
        ).reshape(-1).astype(np.uint32)

        bounds_min = pos_data.min(axis=0)
        bounds_max = pos_data.max(axis=0)
        center = (bounds_min + bounds_max) / 2.0
        pos_data = pos_data - center
        self._model_extent = bounds_max - bounds_min
        longitudinal_data = (
            (pos_data[:, 0] / self._model_extent[0]) + 0.5
        ).reshape(-1, 1)
        
        # Interleave position, normal, UV, and normalized length coordinate.
        vertex_data = np.hstack([
            pos_data.astype(np.float32),
            norm_data.astype(np.float32),
            uv_data.astype(np.float32),
            longitudinal_data.astype(np.float32),
        ]).flatten()
        
        self._vertex_count = len(pos_data)
        self._index_count = len(idx_data)
        
        # Create buffers
        self.vbo = self.ctx.buffer(vertex_data.tobytes())
        self.ibo = self.ctx.buffer(idx_data.tobytes())
        
        # Create VAO
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (
                    self.vbo,
                    '3f 3f 2f 1f',
                    'in_position',
                    'in_normal',
                    'in_uv',
                    'in_longitudinal',
                )
            ],
            self.ibo
        )

    def _create_texture(self, image):
        if image.ndim == 2:
            components = 1
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            components = 4
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            components = 3

        image = cv2.flip(image, 0)
        texture = self.ctx.texture(
            image.shape[:2][::-1], components, image.tobytes(), alignment=1
        )
        texture.build_mipmaps()
        texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        texture.repeat_x = True
        texture.repeat_y = True
        return texture

    def _load_textures(self, gltf, buffer_data):
        """Load textures from GLB."""
        for name, source_index in self._material_texture_sources(gltf).items():
            img = gltf.images[source_index]
            if img.bufferView is None:
                raise ValueError(f"External GLB texture URIs are unsupported: {img.uri}")
            bv = gltf.bufferViews[img.bufferView]
            offset = bv.byteOffset or 0
            tex_data = buffer_data[offset:offset + bv.byteLength]
            nparr = np.frombuffer(tex_data, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img_cv is None:
                raise ValueError(f"Could not decode GLB {name} texture")
            self.textures[name] = self._create_texture(img_cv)

    @staticmethod
    def _material_texture_sources(gltf):
        """Resolve material slots through glTF texture-to-image references."""
        primitive = gltf.meshes[0].primitives[0]
        material = gltf.materials[primitive.material]
        pbr = material.pbrMetallicRoughness
        texture_indices = {
            'albedo': pbr.baseColorTexture.index,
            'metallic_roughness': pbr.metallicRoughnessTexture.index,
            'emissive': material.emissiveTexture.index,
        }
        return {
            name: gltf.textures[texture_index].source
            for name, texture_index in texture_indices.items()
        }

    def _setup_framebuffer(self):
        """Set up offscreen framebuffer for rendering."""
        # We'll create this dynamically based on frame size
        self.fbo = None
        self.fbo_texture = None
        self.fbo_depth = None

    def _ensure_framebuffer(self, width, height):
        """Ensure framebuffer matches frame size."""
        if self.fbo is None or self.fbo.width != width or self.fbo.height != height:
            if self.fbo:
                self.fbo.release()
                self.fbo_texture.release()
                self.fbo_depth.release()
            
            self.fbo_texture = self.ctx.texture((width, height), 4)
            self.fbo_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.fbo_depth = self.ctx.depth_texture((width, height))
            self.fbo = self.ctx.framebuffer(color_attachments=[self.fbo_texture], depth_attachment=self.fbo_depth)

    def set_view_projection(self, width, height):
        """Map model coordinates to the exact video-frame pixel space."""
        self.view_matrix = Matrix44.identity()
        self.projection_matrix = Matrix44.orthogonal_projection(
            0.0, float(width), 0.0, float(height), -100.0, 100.0
        )

    def update_glow(self, should_glow):
        """Update glow target state."""
        self.target_emissive_intensity = 1.0 if should_glow else 0.0

    def _update_glow_animation(self):
        """Animate glow intensity."""
        if self.emissive_intensity < self.target_emissive_intensity:
            self.emissive_intensity = min(self.target_emissive_intensity, 
                                         self.emissive_intensity + self.fade_in_speed)
        elif self.emissive_intensity > self.target_emissive_intensity:
            self.emissive_intensity = max(self.target_emissive_intensity, 
                                         self.emissive_intensity - self.fade_out_speed)

    def _model_matrix(self, frame_height, cigarette_tracker):
        """Build a pixel-aligned transform from the tracked cigarette pose."""
        pixel_scale = cigarette_tracker.length / self._model_extent[0]
        pixel_scale *= self.model_scale
        pos_2d = cigarette_tracker.position
        world_pos = np.array([
            pos_2d[0] + self.model_offset[0],
            frame_height - pos_2d[1] - self.model_offset[1],
            self.model_offset[2],
        ], dtype=np.float32)

        model_matrix = Matrix44.from_translation(world_pos)
        model_matrix *= Matrix44.from_z_rotation(
            cigarette_tracker.rotation + self.rotation_offset[2]
        )
        model_matrix *= Matrix44.from_x_rotation(self.rotation_offset[0])
        model_matrix *= Matrix44.from_y_rotation(self.rotation_offset[1])
        model_matrix *= Matrix44.from_scale([pixel_scale, pixel_scale, pixel_scale])
        return model_matrix

    def render(self, frame, cigarette_tracker, face_tracker=None):
        """
        Render 3D cigarette onto the frame.
        
        Args:
            frame: OpenCV frame (BGR)
            cigarette_tracker: CigaretteTracker instance with position/rotation
            face_tracker: FaceTracker for mouth position (optional)
        """
        if not cigarette_tracker.is_held or cigarette_tracker.position is None:
            return frame
        
        h, w = frame.shape[:2]
        self._ensure_framebuffer(w, h)
        
        # Update glow animation
        self._update_glow_animation()
        
        if self._model_extent[0] <= 0:
            return frame
        model_matrix = self._model_matrix(h, cigarette_tracker)
        
        # Render to offscreen framebuffer
        self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        
        # Set uniforms
        self.program['model'].write(model_matrix.astype('f4').tobytes())
        self.program['view'].write(self.view_matrix.astype('f4').tobytes())
        self.program['projection'].write(self.projection_matrix.astype('f4').tobytes())
        self.program['emissive_intensity'].write(np.float32(self.emissive_intensity))
        self.program['use_emissive'].write(np.int32(1 if self.emissive_intensity > 0.01 else 0))
        
        # Bind textures
        self.textures['albedo'].use(0)
        self.program['albedo_map'].value = 0
        self.textures['metallic_roughness'].use(1)
        self.program['metallic_roughness_map'].value = 1
        self.textures['emissive'].use(2)
        self.program['emissive_map'].value = 2
        
        # Draw
        self.vao.render(moderngl.TRIANGLES)
        
        # Read framebuffer to numpy array
        data = self.fbo.read(components=4, alignment=1)
        rendered = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
        rendered = cv2.flip(rendered, 0)  # Flip back (OpenGL origin bottom-left)
        rendered = cv2.cvtColor(rendered, cv2.COLOR_RGBA2BGRA)
        
        # Alpha blend onto frame
        return self._alpha_blend(frame, rendered)

    def _alpha_blend(self, frame, overlay):
        """Alpha blend overlay onto frame."""
        if overlay is None or overlay.size == 0:
            return frame
        
        # overlay is BGRA
        alpha = overlay[:, :, 3:4] / 255.0
        alpha = np.clip(alpha, 0, 1)
        
        # Blend
        frame_float = frame.astype(np.float32)
        overlay_rgb = overlay[:, :, :3].astype(np.float32)
        
        blended = frame_float * (1 - alpha) + overlay_rgb * alpha
        return blended.astype(np.uint8)

    def get_debug_info(self):
        return {
            'emissive_intensity': self.emissive_intensity,
            'target_emissive': self.target_emissive_intensity,
            'vertex_count': self._vertex_count,
            'index_count': self._index_count,
            'model_extent': self._model_extent.copy(),
        }

    def close(self):
        if self.vao:
            self.vao.release()
        if self.vbo:
            self.vbo.release()
        if self.ibo:
            self.ibo.release()
        if self.program:
            self.program.release()
        for tex in self.textures.values():
            tex.release()
        if self.fbo:
            self.fbo.release()
        if self.fbo_texture:
            self.fbo_texture.release()
        if self.fbo_depth:
            self.fbo_depth.release()
        if self.ctx:
            self.ctx.release()


class Cigarette3DRendererFallback:
    """Fallback that uses the original 2D renderer if 3D fails."""
    def __init__(self):
        from effects.cigarette import CigaretteRendererFallback
        self.fallback = CigaretteRendererFallback()
    
    def render(self, frame, cigarette_tracker, face_tracker=None):
        if cigarette_tracker.is_held and cigarette_tracker.position is not None:
            self.fallback.draw(frame, cigarette_tracker.position, cigarette_tracker.rotation, 0.0)
        return frame
    
    def update_glow(self, should_glow):
        pass
    
    def get_debug_info(self):
        return {'fallback': True}
    
    def close(self):
        pass
