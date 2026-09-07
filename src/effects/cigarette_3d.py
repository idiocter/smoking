import moderngl
import numpy as np
import pygltflib
import cv2
import os
from pyrr import Matrix44, Vector3, Quaternion


VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 v_position;
out vec3 v_normal;
out vec2 v_uv;

void main() {
    v_position = (model * vec4(in_position, 1.0)).xyz;
    v_normal = mat3(transpose(inverse(model))) * in_normal;
    v_uv = in_uv;
    gl_Position = projection * view * model * vec4(in_position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec3 v_position;
in vec3 v_normal;
in vec2 v_uv;

uniform sampler2D albedo_map;
uniform sampler2D metallic_roughness_map;
uniform sampler2D normal_map;
uniform sampler2D emissive_map;

uniform vec3 light_pos;
uniform vec3 view_pos;
uniform vec3 light_color;
uniform float emissive_intensity;
uniform int use_emissive;

out vec4 frag_color;

void main() {
    vec3 albedo = texture(albedo_map, v_uv).rgb;
    vec3 metallic_roughness = texture(metallic_roughness_map, v_uv).rgb;
    float metallic = metallic_roughness.b;
    float roughness = metallic_roughness.g;
    
    // Use vertex normal directly (skip normal map for now to avoid tangent space issues)
    vec3 N = normalize(v_normal);
    
    // Lighting
    vec3 L = normalize(light_pos - v_position);
    vec3 V = normalize(view_pos - v_position);
    vec3 H = normalize(L + V);
    
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    
    // Simplified PBR
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float NDF = 1.0 / (3.14159 * roughness * roughness * pow(NdotH * NdotH * (1.0 - k) + k, 2.0));
    float G = NdotL / (NdotL * (1.0 - k) + k);
    vec3 F = vec3(0.04) + (1.0 - vec3(0.04)) * pow(1.0 - NdotH, 5.0);
    
    vec3 diffuse = albedo / 3.14159 * (1.0 - metallic);
    vec3 specular = F * NDF * G / (4.0 * NdotL + 0.001);
    
    vec3 lighting = (diffuse + specular) * light_color * NdotL;
    
    // Emissive (glow)
    vec3 emissive = vec3(0.0);
    if (use_emissive != 0) {
        vec3 emissive_tex = texture(emissive_map, v_uv).rgb;
        emissive = emissive_tex * emissive_intensity * vec3(1.0, 0.5, 0.1); // Orange glow
    }
    
    vec3 color = albedo * lighting + emissive;
    
    // Alpha from albedo (assuming opaque)
    float alpha = 1.0;
    
    frag_color = vec4(color, alpha);
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
        
        self._init_gl()
        self._load_model(model_path)
        self._load_textures(model_path)
        self._setup_framebuffer()
        
        # Camera setup (will be updated per frame)
        self.view_matrix = Matrix44.identity()
        self.projection_matrix = Matrix44.identity()
        self.light_pos = Vector3([2.0, 2.0, 2.0])
        self.view_pos = Vector3([0.0, 0.0, 3.0])
        self.light_color = Vector3([1.0, 1.0, 1.0])
        
        # Glow state
        self.emissive_intensity = 0.0
        self.target_emissive_intensity = 0.0
        self.fade_in_speed = self.config.get('glow_fade_in', 0.15)
        self.fade_out_speed = self.config.get('glow_fade_out', 0.08)

    def _init_gl(self):
        """Initialize ModernGL context for offscreen rendering."""
        # Create a headless context
        self.ctx = moderngl.create_context(standalone=True)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.cull_face = 'back'
        
        # Compile shaders
        self.program = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER
        )

    def _load_model(self, model_path):
        """Load GLB model and extract mesh data."""
        gltf = pygltflib.GLTF2().load(model_path)
        buffer_data = gltf.binary_blob()
        
        mesh = gltf.meshes[0]
        prim = mesh.primitives[0]
        
        # Get accessors
        pos_acc = gltf.accessors[prim.attributes.POSITION]
        norm_acc = gltf.accessors[prim.attributes.NORMAL]
        uv_acc = gltf.accessors[prim.attributes.TEXCOORD_0]
        idx_acc = gltf.accessors[prim.indices]
        
        # Get buffer views
        pos_bv = gltf.bufferViews[pos_acc.bufferView]
        norm_bv = gltf.bufferViews[norm_acc.bufferView]
        uv_bv = gltf.bufferViews[uv_acc.bufferView]
        idx_bv = gltf.bufferViews[idx_acc.bufferView]
        
        # Extract data
        pos_data = np.frombuffer(
            buffer_data[pos_bv.byteOffset + pos_acc.byteOffset:
                       pos_bv.byteOffset + pos_acc.byteOffset + pos_acc.count * 12],
            dtype=np.float32).reshape(-1, 3)
        norm_data = np.frombuffer(
            buffer_data[norm_bv.byteOffset + norm_acc.byteOffset:
                       norm_bv.byteOffset + norm_acc.byteOffset + norm_acc.count * 12],
            dtype=np.float32).reshape(-1, 3)
        uv_data = np.frombuffer(
            buffer_data[uv_bv.byteOffset + uv_acc.byteOffset:
                       uv_bv.byteOffset + uv_acc.byteOffset + uv_acc.count * 8],
            dtype=np.float32).reshape(-1, 2)
        idx_data = np.frombuffer(
            buffer_data[idx_bv.byteOffset + idx_acc.byteOffset:
                       idx_bv.byteOffset + idx_acc.byteOffset + idx_acc.count * 4],
            dtype=np.uint32).reshape(-1)
        
        # Center the model
        center = pos_data.mean(axis=0)
        pos_data = pos_data - center
        
        # Scale model
        scale = self.model_scale
        pos_data = pos_data * scale
        
        # Apply model offset
        pos_data = pos_data + self.model_offset
        
        # Interleave vertex data: position(3), normal(3), uv(2) = 8 floats per vertex
        vertex_data = np.hstack([
            pos_data.astype(np.float32),
            norm_data.astype(np.float32),
            uv_data.astype(np.float32)
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
                (self.vbo, '3f 3f 2f', 'in_position', 'in_normal', 'in_uv')
            ],
            self.ibo
        )

    def _load_textures(self, model_path):
        """Load textures from GLB."""
        gltf = pygltflib.GLTF2().load(model_path)
        buffer_data = gltf.binary_blob()
        
        # Material has: baseColorTexture=0, metallicRoughnessTexture=1
        # We'll use texture 0 for albedo, texture 1 for metallic/roughness
        # Texture 2 for normal, texture 3 for emissive
        for tex_idx in range(4):
            img = gltf.images[tex_idx]
            bv = gltf.bufferViews[img.bufferView]
            tex_data = buffer_data[bv.byteOffset:bv.byteOffset + bv.byteLength]
            
            # Decode image
            nparr = np.frombuffer(tex_data, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if img_cv is not None:
                # Convert BGR to RGB
                if len(img_cv.shape) == 3:
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                # Flip vertically for OpenGL
                img_cv = cv2.flip(img_cv, 0)
                
                # Create texture
                texture = self.ctx.texture(img_cv.shape[:2][::-1], 
                                          3 if len(img_cv.shape) == 3 else 1,
                                          img_cv.tobytes())
                texture.build_mipmaps()
                texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
                texture.repeat_x = True
                texture.repeat_y = True
                self.textures[tex_idx] = texture
            else:
                print(f"Warning: Failed to decode texture {tex_idx}")

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

    def set_view_projection(self, width, height, fov=60.0, near=0.1, far=100.0):
        """Update view and projection matrices based on camera parameters."""
        aspect = width / height
        self.projection_matrix = Matrix44.perspective_projection(fov, aspect, near, far)
        
        # View matrix - camera at origin looking at -Z
        self.view_matrix = Matrix44.look_at(
            self.view_pos,  # eye
            Vector3([0.0, 0.0, 0.0]),  # center
            Vector3([0.0, 1.0, 0.0])   # up
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
        
        # Get cigarette position and rotation from tracker
        # Tracker gives 2D position and rotation in image coordinates
        # We need to convert to 3D world coordinates
        pos_2d = cigarette_tracker.position
        rot_2d = cigarette_tracker.rotation
        
        # Convert 2D position to 3D world position
        # Use simple projection: assume camera at origin, looking at -Z
        # Image coordinates: (0,0) top-left, (w,h) bottom-right
        # Normalize to [-1, 1] then map to world
        norm_x = (pos_2d[0] / w) * 2.0 - 1.0
        norm_y = 1.0 - (pos_2d[1] / h) * 2.0  # Flip Y
        
        # Estimate depth from hand size / distance to mouth
        # Simple heuristic: use fixed depth with some variation
        depth = -1.5  # 1.5 units in front of camera (closer)
        
        # World position
        world_pos = Vector3([norm_x * 2.0, norm_y * 2.0, depth])
        
        # Apply model offset
        world_pos = world_pos + self.model_offset
        
        # Rotation: 2D rotation maps to Y-axis rotation (around vertical)
        # plus some X rotation for tilt
        rot_y = -rot_2d + self.rotation_offset[1]  # Negative because image coords
        rot_x = self.rotation_offset[0]  # Fixed tilt
        rot_z = self.rotation_offset[2]
        
        # Build model matrix
        model_matrix = Matrix44.identity()
        model_matrix = model_matrix * Matrix44.from_translation(world_pos)
        model_matrix = model_matrix * Matrix44.from_eulers([rot_x, rot_y, rot_z])
        
        # Render to offscreen framebuffer
        self.fbo.use()
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        
        # Set uniforms
        self.program['model'].write(model_matrix.astype('f4').tobytes())
        self.program['view'].write(self.view_matrix.astype('f4').tobytes())
        self.program['projection'].write(self.projection_matrix.astype('f4').tobytes())
        self.program['light_pos'].write(self.light_pos.astype('f4'))
        self.program['view_pos'].write(self.view_pos.astype('f4'))
        self.program['light_color'].write(self.light_color.astype('f4'))
        self.program['emissive_intensity'].write(np.float32(self.emissive_intensity))
        self.program['use_emissive'].write(np.int32(1 if self.emissive_intensity > 0.01 else 0))
        
        # Bind textures
        if 0 in self.textures:
            self.textures[0].use(0)
            self.program['albedo_map'].value = 0
        if 1 in self.textures:
            self.textures[1].use(1)
            self.program['metallic_roughness_map'].value = 1
        # normal_map removed from shader
        if 3 in self.textures:
            self.textures[3].use(3)
            self.program['emissive_map'].value = 3
        
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