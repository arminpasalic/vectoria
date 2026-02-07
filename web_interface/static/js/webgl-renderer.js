/**
 * WebGL-accelerated visualization renderer for RAG-Vectoria
 * Provides hardware acceleration for improved performance with large datasets
 */

class WebGLRenderer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.gl = null;
        this.program = null;
        this.buffers = {};
        this.uniforms = {};
        this.attributes = {};
        
        // Configuration
        this.maxPoints = options.maxPoints || 100000;
        this.pointSizeRange = options.pointSizeRange || [2, 12];
        this.enableOutlines = options.enableOutlines || true;
        
        // WebGL state
        this.isInitialized = false;
        this.needsRedraw = true;
        
        // Data storage
        this.pointData = null;
        this.colorData = null;
        this.sizeData = null;
        this.outlineData = null;
        this.coordIndexMap = new Map(); // "x,y" -> index
        this.docIndexMap = new Map();   // doc_id -> index
        
        // Fallback to Canvas 2D if WebGL not available
        this.useFallback = false;
        
        this.initialize();
    }
    
    initialize() {
        try {
            // Check if WebGL is available at all
            if (!this.isWebGLSupported()) {
                // WebGL not available, use Canvas 2D (no warning needed)
                this.useFallback = true;
                return;
            }
            
            // Try creating a context with progressively relaxed options
            const base = {
                alpha: true,
                antialias: true,
                depth: false,
                stencil: false,
                preserveDrawingBuffer: false,
                powerPreference: 'high-performance',
                failIfMajorPerformanceCaveat: false,
                desynchronized: true
            };
            const variants = [
                base,
                { ...base, antialias: false },
                { ...base, alpha: false },
                { ...base, antialias: false, alpha: false },
                { ...base, desynchronized: false },
            ];

            this.gl = this.tryCreateContext('webgl2', variants) ||
                      this.tryCreateContext('webgl', variants) ||
                      this.tryCreateContext('experimental-webgl', variants);
            
            if (!this.gl) {
                // This is expected when canvas already has a 2D context
                // Silently fall back to Canvas 2D (which is already being used)
                this.useFallback = true;
                return;
            }
            
            // Test basic WebGL functionality
            if (!this.testWebGLCapabilities()) {
                // WebGL capabilities insufficient, fallback to Canvas 2D
                this.useFallback = true;
                return;
            }
            
            this.setupShaders();
            this.setupBuffers();
            this.setupUniforms();
            this.isInitialized = true;
            this.setupContextLossHandlers();
            
        } catch (error) {
            // Silent fallback to Canvas 2D - this is expected behavior
            this.useFallback = true;
        }
    }

    tryCreateContext(type, optionsList) {
        for (const opts of optionsList) {
            try {
                const gl = this.canvas.getContext(type, opts);
                if (gl) return gl;
            } catch (e) {
                // ignore and try next
            }
        }
        return null;
    }
    
    isWebGLSupported() {
        try {
            const canvas = document.createElement('canvas');
            return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
        } catch (e) {
            return false;
        }
    }
    
    testWebGLCapabilities() {
        if (!this.gl) return false;
        
        try {
            // Test basic WebGL operations
            this.gl.getExtension('OES_vertex_array_object');
            const maxVertexAttribs = this.gl.getParameter(this.gl.MAX_VERTEX_ATTRIBS);
            const maxTextureSize = this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE);
            
            // Ensure minimum required capabilities
            return maxVertexAttribs >= 8 && maxTextureSize >= 256;
        } catch (error) {
            return false;
        }
    }

    setupContextLossHandlers() {
        if (!this.gl) return;
        this.canvas.addEventListener('webglcontextlost', (e) => {
            e.preventDefault();
            console.warn('⚠️ WebGL context lost — falling back to Canvas temporarily');
            this.useFallback = true;
        }, false);
        this.canvas.addEventListener('webglcontextrestored', () => {
            this.useFallback = false;
            try {
                this.setupShaders();
                this.setupBuffers();
                this.setupUniforms();
                this.isInitialized = true;
                if (this.pointData) {
                    // Re-upload data
                    this.loadData(Array.from({ length: this.pointData.count }).map((_, i) => ({
                        x: this.pointData.positions[i * 2],
                        y: this.pointData.positions[i * 2 + 1],
                        color: `rgba(${this.pointData.colors[i * 4]}, ${this.pointData.colors[i * 4 + 1]}, ${this.pointData.colors[i * 4 + 2]}, ${this.pointData.colors[i * 4 + 3] / 255})`,
                        size: this.pointData.sizes[i],
                        opacity: this.pointData.opacities[i]
                    })));
                }
            } catch (e) {
                console.error('Failed to restore WebGL context:', e);
                this.useFallback = true;
            }
        }, false);
    }
    
    setupShaders() {
        // Shader for rendering points
        const vertexShaderSource = `
            attribute vec2 a_position;
            attribute vec4 a_color;
            attribute float a_size;
            attribute float a_opacity;

            uniform mat3 u_transform;
            uniform vec2 u_resolution;
            uniform float u_pointScale;

            varying vec4 v_color;
            varying float v_size;

            void main() {
                // Apply transformation
                vec3 position = u_transform * vec3(a_position, 1.0);

                // Convert to clip space
                vec2 clipSpace = ((position.xy / u_resolution) * 2.0 - 1.0) * vec2(1, -1);
                gl_Position = vec4(clipSpace, 0.0, 1.0);

                // Pass color with opacity
                v_color = vec4(a_color.rgb, a_color.a * a_opacity);

                // Set point size
                v_size = a_size * u_pointScale;
                gl_PointSize = v_size;
            }
        `;

        const fragmentShaderSource = `
            precision mediump float;

            varying vec4 v_color;
            varying float v_size;

            uniform bool u_enableCircles;
            uniform bool u_enableOutlines;
            uniform vec4 u_outlineColor;
            uniform float u_outlineWidth;

            void main() {
                if (u_enableCircles) {
                    // Calculate distance from center
                    vec2 center = gl_PointCoord - vec2(0.5);
                    float dist = length(center);

                    // Create circular points with anti-aliasing
                    if (dist > 0.5) {
                        discard;
                    }

                    // Add subtle outline effect (stroke)
                    if (u_enableOutlines && dist > (0.5 - u_outlineWidth / v_size)) {
                        // Subtle white/light gray stroke with transparency
                        gl_FragColor = u_outlineColor;
                    } else {
                        // Smooth anti-aliased edges with full vibrancy
                        float baseAlpha = v_color.a; // Full opacity for vibrant colors
                        float edgeSmooth = 1.0 - smoothstep(0.42, 0.5, dist);
                        gl_FragColor = vec4(v_color.rgb, baseAlpha * edgeSmooth);
                    }
                } else {
                    gl_FragColor = v_color;
                }
            }
        `;

        // Shader for rendering perspective grid
        const gridVertexShaderSource = `
            attribute vec2 a_position;
            uniform vec2 u_resolution;
            varying vec2 v_worldPos;

            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                // Pass world position to fragment shader
                v_worldPos = (a_position + 1.0) * 0.5 * u_resolution;
            }
        `;

        const gridFragmentShaderSource = `
            precision mediump float;

            varying vec2 v_worldPos;
            uniform vec2 u_resolution;
            uniform mat3 u_transform;
            uniform bool u_isDarkMode;

            void main() {
                // Inverse transform to get world coordinates
                mat3 invTransform = mat3(
                    1.0 / u_transform[0][0], 0.0, -u_transform[0][2] / u_transform[0][0],
                    0.0, 1.0 / u_transform[1][1], -u_transform[1][2] / u_transform[1][1],
                    0.0, 0.0, 1.0
                );

                vec3 worldPos3 = invTransform * vec3(v_worldPos, 1.0);
                vec2 worldPos = worldPos3.xy;

                // Grid parameters - larger spacing for subtlety
                float gridSpacing = 150.0;

                // Simple grid calculation without perspective
                vec2 grid = abs(fract(worldPos / gridSpacing - 0.5) - 0.5) / fwidth(worldPos / gridSpacing);
                float line = min(grid.x, grid.y);

                // Fade based on distance from center for depth feel
                vec2 center = u_resolution * 0.5;
                vec2 fromCenter = v_worldPos - center;
                float distFromCenter = length(fromCenter);
                float fadeFactor = 1.0 - smoothstep(0.0, u_resolution.x * 0.7, distFromCenter);

                // Color based on theme - subtle but visible
                vec3 gridColor;
                float baseAlpha;
                if (u_isDarkMode) {
                    gridColor = vec3(0.18, 0.20, 0.24); // Subtle lighter gray for dark mode
                    baseAlpha = 0.25;
                } else {
                    gridColor = vec3(0.85, 0.86, 0.88); // Subtle darker gray for light mode
                    baseAlpha = 0.35;
                }

                float alpha = (1.0 - min(line, 1.0)) * baseAlpha * fadeFactor;
                gl_FragColor = vec4(gridColor, alpha);
            }
        `;

        // Compile point shaders
        const vertexShader = this.compileShader(vertexShaderSource, this.gl.VERTEX_SHADER);
        const fragmentShader = this.compileShader(fragmentShaderSource, this.gl.FRAGMENT_SHADER);

        // Create point program
        this.program = this.gl.createProgram();
        this.gl.attachShader(this.program, vertexShader);
        this.gl.attachShader(this.program, fragmentShader);
        this.gl.linkProgram(this.program);

        if (!this.gl.getProgramParameter(this.program, this.gl.LINK_STATUS)) {
            throw new Error('Shader program linking failed: ' + this.gl.getProgramInfoLog(this.program));
        }

        // Compile grid shaders
        const gridVertexShader = this.compileShader(gridVertexShaderSource, this.gl.VERTEX_SHADER);
        const gridFragmentShader = this.compileShader(gridFragmentShaderSource, this.gl.FRAGMENT_SHADER);

        // Create grid program
        this.gridProgram = this.gl.createProgram();
        this.gl.attachShader(this.gridProgram, gridVertexShader);
        this.gl.attachShader(this.gridProgram, gridFragmentShader);
        this.gl.linkProgram(this.gridProgram);

        if (!this.gl.getProgramParameter(this.gridProgram, this.gl.LINK_STATUS)) {
            throw new Error('Grid shader program linking failed: ' + this.gl.getProgramInfoLog(this.gridProgram));
        }
    }
    
    compileShader(source, type) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            const error = this.gl.getShaderInfoLog(shader);
            this.gl.deleteShader(shader);
            throw new Error('Shader compilation failed: ' + error);
        }
        
        return shader;
    }
    
    setupBuffers() {
        // Position buffer for points
        this.buffers.position = this.gl.createBuffer();

        // Color buffer
        this.buffers.color = this.gl.createBuffer();

        // Size buffer
        this.buffers.size = this.gl.createBuffer();

        // Opacity buffer
        this.buffers.opacity = this.gl.createBuffer();

        // Grid buffer - fullscreen quad
        this.buffers.grid = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.grid);
        const gridVertices = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
            -1,  1,
             1, -1,
             1,  1
        ]);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, gridVertices, this.gl.STATIC_DRAW);
    }
    
    setupUniforms() {
        // Point shader uniforms
        this.gl.useProgram(this.program);

        // Get attribute locations
        this.attributes.position = this.gl.getAttribLocation(this.program, 'a_position');
        this.attributes.color = this.gl.getAttribLocation(this.program, 'a_color');
        this.attributes.size = this.gl.getAttribLocation(this.program, 'a_size');
        this.attributes.opacity = this.gl.getAttribLocation(this.program, 'a_opacity');

        // Get uniform locations
        this.uniforms.transform = this.gl.getUniformLocation(this.program, 'u_transform');
        this.uniforms.resolution = this.gl.getUniformLocation(this.program, 'u_resolution');
        this.uniforms.pointScale = this.gl.getUniformLocation(this.program, 'u_pointScale');
        this.uniforms.enableCircles = this.gl.getUniformLocation(this.program, 'u_enableCircles');
        this.uniforms.enableOutlines = this.gl.getUniformLocation(this.program, 'u_enableOutlines');
        this.uniforms.outlineColor = this.gl.getUniformLocation(this.program, 'u_outlineColor');
        this.uniforms.outlineWidth = this.gl.getUniformLocation(this.program, 'u_outlineWidth');

        // Grid shader uniforms
        this.gl.useProgram(this.gridProgram);
        this.gridUniforms = {
            resolution: this.gl.getUniformLocation(this.gridProgram, 'u_resolution'),
            transform: this.gl.getUniformLocation(this.gridProgram, 'u_transform'),
            isDarkMode: this.gl.getUniformLocation(this.gridProgram, 'u_isDarkMode')
        };
        this.gridAttributes = {
            position: this.gl.getAttribLocation(this.gridProgram, 'a_position')
        };
    }
    
    loadData(points) {
        if (this.useFallback) return;
        
        if (!points || points.length === 0) {
            this.pointData = null;
            return;
        }
        
        // Disable outlines automatically for very large datasets to improve performance
        if (points.length > 500000) {
            this.enableOutlines = false;
        }
        
        // Prepare data arrays (compact color to bytes for memory efficiency)
        const positions = new Float32Array(points.length * 2);
        const colors = new Uint8Array(points.length * 4); // normalized in shader
        const sizes = new Float32Array(points.length);
        const opacities = new Float32Array(points.length);
        const baseColors = new Uint8Array(points.length * 4);
        const baseOpacities = new Float32Array(points.length);
        const outlierMask = new Uint8Array(points.length);

        this.coordIndexMap.clear();
        this.docIndexMap.clear();

        for (let i = 0; i < points.length; i++) {
            const point = points[i];
            const idx2 = i * 2;
            const idx4 = i * 4;
            
            // Position
            positions[idx2] = point.x || 0;
            positions[idx2 + 1] = point.y || 0;
            this.coordIndexMap.set(`${positions[idx2]},${positions[idx2 + 1]}`, i);
            if (point.doc_id !== undefined && point.doc_id !== null) {
                this.docIndexMap.set(point.doc_id, i);
            }
            
            // Color (convert hex to RGBA bytes)
            const color = this.hexToRgbaBytes(point.color || '#3498DB');
            colors[idx4] = color.r;
            colors[idx4 + 1] = color.g;
            colors[idx4 + 2] = color.b;
            colors[idx4 + 3] = color.a || 255;
            baseColors[idx4] = colors[idx4];
            baseColors[idx4 + 1] = colors[idx4 + 1];
            baseColors[idx4 + 2] = colors[idx4 + 2];
            baseColors[idx4 + 3] = colors[idx4 + 3];
            
            // Size
            sizes[i] = point.size || 4;
            
            // Opacity
            const baseOpacity = point.cluster === -1 ? 0.2 : (point.opacity !== undefined ? point.opacity : 1.0);
            opacities[i] = baseOpacity;
            baseOpacities[i] = baseOpacity;
            if (point.cluster === -1) {
                outlierMask[i] = 1;
            }
        }

        // Upload to GPU
        this.uploadBuffer(this.buffers.position, positions, 2, this.attributes.position, this.gl.FLOAT, false, this.gl.STATIC_DRAW);
        // Colors as normalized unsigned bytes
        this.uploadBuffer(this.buffers.color, colors, 4, this.attributes.color, this.gl.UNSIGNED_BYTE, true, this.gl.STATIC_DRAW);
        this.uploadBuffer(this.buffers.size, sizes, 1, this.attributes.size, this.gl.FLOAT, false, this.gl.STATIC_DRAW);
        // Opacities change frequently; use DYNAMIC_DRAW
        this.uploadBuffer(this.buffers.opacity, opacities, 1, this.attributes.opacity, this.gl.FLOAT, false, this.gl.DYNAMIC_DRAW);
        
        this.pointData = {
            count: points.length,
            positions,
            colors,
            baseColors,
            sizes,
            opacities,
            baseOpacities,
            isOutlier: outlierMask
        };
        
        this.needsRedraw = true;
    }
    
    uploadBuffer(buffer, data, itemSize, attribute, type, normalized, usageHint) {
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, data, usageHint || this.gl.STATIC_DRAW);
        
        this.gl.enableVertexAttribArray(attribute);
        this.gl.vertexAttribPointer(attribute, itemSize, type || this.gl.FLOAT, !!normalized, 0, 0);
    }
    
    render(transform, viewport) {
        if (this.useFallback || !this.isInitialized || !this.pointData) {
            return;
        }

        const { width, height } = viewport;

        // Set viewport
        this.gl.viewport(0, 0, width, height);

        // Clear with appropriate background color based on theme
        const isDark = document.documentElement.classList.contains('dark');
        if (isDark) {
            this.gl.clearColor(0.067, 0.078, 0.106, 1.0); // rgba(17, 20, 27, 1) - dark background
        } else {
            this.gl.clearColor(0.98, 0.98, 0.99, 1.0); // rgba(250, 250, 252, 1) - light background
        }
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        // Enable blending for transparency
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

        // ============ Draw grid background ============
        this.gl.useProgram(this.gridProgram);

        // Set grid uniforms
        this.gl.uniform2f(this.gridUniforms.resolution, width, height);
        this.gl.uniformMatrix3fv(this.gridUniforms.transform, false, transform);
        this.gl.uniform1i(this.gridUniforms.isDarkMode, isDark ? 1 : 0);

        // Bind grid buffer and draw
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.grid);
        this.gl.enableVertexAttribArray(this.gridAttributes.position);
        this.gl.vertexAttribPointer(this.gridAttributes.position, 2, this.gl.FLOAT, false, 0, 0);
        this.gl.drawArrays(this.gl.TRIANGLES, 0, 6);

        // ============ Draw points on top ============
        this.gl.useProgram(this.program);

        // Set uniforms
        this.gl.uniformMatrix3fv(this.uniforms.transform, false, transform);
        this.gl.uniform2f(this.uniforms.resolution, width, height);
        this.gl.uniform1f(this.uniforms.pointScale, 1.0);
        this.gl.uniform1i(this.uniforms.enableCircles, true);
        this.gl.uniform1i(this.uniforms.enableOutlines, this.enableOutlines);
        // Subtle stroke: white with 30% opacity (rgba(255, 255, 255, 0.3))
        this.gl.uniform4f(this.uniforms.outlineColor, 1.0, 1.0, 1.0, 0.3);
        this.gl.uniform1f(this.uniforms.outlineWidth, 1.0);

        // Bind buffers and set attributes
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.position);
        this.gl.enableVertexAttribArray(this.attributes.position);
        this.gl.vertexAttribPointer(this.attributes.position, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.color);
        this.gl.enableVertexAttribArray(this.attributes.color);
        // Colors were uploaded as UNSIGNED_BYTE, normalized
        this.gl.vertexAttribPointer(this.attributes.color, 4, this.gl.UNSIGNED_BYTE, true, 0, 0);

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.size);
        this.gl.enableVertexAttribArray(this.attributes.size);
        this.gl.vertexAttribPointer(this.attributes.size, 1, this.gl.FLOAT, false, 0, 0);

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.opacity);
        this.gl.enableVertexAttribArray(this.attributes.opacity);
        this.gl.vertexAttribPointer(this.attributes.opacity, 1, this.gl.FLOAT, false, 0, 0);

        // Draw points
        this.gl.drawArrays(this.gl.POINTS, 0, this.pointData.count);

        this.needsRedraw = false;
    }
    
    updateHighlighting(highlightedIndices, searchResults) {
        if (this.useFallback || !this.pointData) return;

        const count = this.pointData.count;
        const baseOpacities = this.pointData.baseOpacities || new Float32Array(count).fill(1.0);
        const baseColors = this.pointData.baseColors || this.pointData.colors;
        const opacities = new Float32Array(baseOpacities);
        const colors = new Uint8Array(baseColors);

        let dimOthers = false;
        const toHighlight = new Set();

        if (Array.isArray(searchResults) && searchResults.length > 0) {
            dimOthers = true;
            for (const r of searchResults) {
                if (typeof r.index === 'number' && r.index >= 0 && r.index < count) {
                    toHighlight.add(r.index);
                    continue;
                }
                if (r.coordinates && r.coordinates.length >= 2) {
                    const key = `${r.coordinates[0]},${r.coordinates[1]}`;
                    const idx = this.coordIndexMap.get(key);
                    if (typeof idx === 'number') {
                        toHighlight.add(idx);
                        continue;
                    }
                }
                if (r.doc_id !== undefined && r.doc_id !== null) {
                    const idx = this.docIndexMap.get(r.doc_id);
                    if (typeof idx === 'number') toHighlight.add(idx);
                }
            }
        }

        if (Array.isArray(highlightedIndices)) {
            for (const idx of highlightedIndices) {
                if (typeof idx === 'number' && idx >= 0 && idx < count) toHighlight.add(idx);
            }
        }

        if (!(dimOthers || toHighlight.size > 0)) {
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.opacity);
            this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, baseOpacities);
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.color);
            this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, baseColors);
            this.pointData.opacities = new Float32Array(baseOpacities);
            this.pointData.colors = new Uint8Array(baseColors);
            this.needsRedraw = true;
            return;
        }

        const dimOpacity = 0.4;
        const dimRGB = [118, 122, 130];

        for (let i = 0; i < count; i++) {
            if (toHighlight.has(i)) {
                opacities[i] = 1.0;
                const baseOffset = i * 4;
                colors[baseOffset] = baseColors[baseOffset];
                colors[baseOffset + 1] = baseColors[baseOffset + 1];
                colors[baseOffset + 2] = baseColors[baseOffset + 2];
                colors[baseOffset + 3] = baseColors[baseOffset + 3];
                continue;
            }

            if (this.pointData.isOutlier && this.pointData.isOutlier[i]) {
                opacities[i] = baseOpacities[i] || 0.2;
                const baseOffset = i * 4;
                colors[baseOffset] = baseColors[baseOffset];
                colors[baseOffset + 1] = baseColors[baseOffset + 1];
                colors[baseOffset + 2] = baseColors[baseOffset + 2];
                colors[baseOffset + 3] = baseColors[baseOffset + 3];
                continue;
            }

            opacities[i] = dimOpacity;
            const offset = i * 4;
            colors[offset] = dimRGB[0];
            colors[offset + 1] = dimRGB[1];
            colors[offset + 2] = dimRGB[2];
            colors[offset + 3] = 255;
        }

        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.opacity);
        this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, opacities);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.color);
        this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, colors);
        this.pointData.opacities = opacities;
        this.pointData.colors = colors;
        this.needsRedraw = true;
    }

    // Apply metadata filtering mask (visibleIndices is a Set of indices)
    applyMetadataFilter(visibleIndices) {
        if (this.useFallback || !this.pointData) return;
        const count = this.pointData.count;
        const baseOpacities = this.pointData.baseOpacities || new Float32Array(count).fill(1.0);
        const opacities = new Float32Array(baseOpacities);
        if (visibleIndices && visibleIndices.size > 0) {
            opacities.fill(0.2);
            visibleIndices.forEach(idx => {
                if (idx >= 0 && idx < count) opacities[idx] = baseOpacities[idx] || 1.0;
            });
        }
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.opacity);
        this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, opacities);
        this.pointData.opacities = opacities;
        this.needsRedraw = true;
    }
    
    hexToRgba(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16) / 255,
            g: parseInt(result[2], 16) / 255,
            b: parseInt(result[3], 16) / 255,
            a: 1.0
        } : { r: 0.2, g: 0.6, b: 0.9, a: 1.0 };
    }
    
    // Byte version for compact color buffers
    hexToRgbaBytes(hex) {
        if (typeof hex === 'string') {
            const trimmed = hex.trim();
            const hexMatch = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(trimmed);
            if (hexMatch) {
                return {
                    r: parseInt(hexMatch[1], 16),
                    g: parseInt(hexMatch[2], 16),
                    b: parseInt(hexMatch[3], 16),
                    a: 255
                };
            }

            const rgbaMatch = /^rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$/i.exec(trimmed);
            if (rgbaMatch) {
                return {
                    r: Math.round(parseFloat(rgbaMatch[1])),
                    g: Math.round(parseFloat(rgbaMatch[2])),
                    b: Math.round(parseFloat(rgbaMatch[3])),
                    a: Math.round(parseFloat(rgbaMatch[4]) * 255)
                };
            }

            const rgbMatch = /^rgb\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)$/i.exec(trimmed);
            if (rgbMatch) {
                return {
                    r: Math.round(parseFloat(rgbMatch[1])),
                    g: Math.round(parseFloat(rgbMatch[2])),
                    b: Math.round(parseFloat(rgbMatch[3])),
                    a: 255
                };
            }
        }

        return { r: 51, g: 153, b: 230, a: 255 };
    }
    
    resize(width, height) {
        if (this.useFallback) return;
        
        this.canvas.width = width;
        this.canvas.height = height;
        this.needsRedraw = true;
    }
    
    destroy() {
        if (this.useFallback || !this.gl) return;

        // Clean up WebGL resources
        if (this.program) {
            this.gl.deleteProgram(this.program);
        }

        if (this.gridProgram) {
            this.gl.deleteProgram(this.gridProgram);
        }

        Object.values(this.buffers).forEach(buffer => {
            if (buffer) {
                this.gl.deleteBuffer(buffer);
            }
        });

        this.gl = null;
        this.program = null;
        this.gridProgram = null;
        this.buffers = {};
        this.pointData = null;
    }
}

// Enhanced Canvas Visualization with WebGL acceleration
class EnhancedCanvasVisualization extends CanvasVisualization {
    constructor(canvasId, tooltipId, options = {}) {
        super(canvasId, tooltipId);
        
        // WebGL renderer
        this.webglRenderer = new WebGLRenderer(this.canvas, {
            maxPoints: options.maxPoints || 100000,
            enableOutlines: options.enableOutlines !== false
        });
        
        this.useWebGL = !this.webglRenderer.useFallback && options.useWebGL !== false;
        
        // Only log if WebGL is successfully enabled (don't spam console with fallback messages)
        if (this.useWebGL) {
        }
    }
    
    loadData(points) {
        super.loadData(points);
        
        if (this.useWebGL) {
            // Prepare data for WebGL
            const webglPoints = points.map((point, index) => ({
                x: point.x,
                y: point.y,
                color: this.getClusterColor(point.cluster),
                size: 4,
                opacity: 1.0,
                index: index,
                doc_id: point.doc_id,
                cluster: point.cluster
            }));
            
            this.webglRenderer.loadData(webglPoints);
        }
    }
    
    render() {
        if (this.useWebGL) {
            this.renderWebGL();
        } else {
            super.render();
        }
    }
    
    renderWebGL() {
        const viewWidth = this.canvas.width / this.dpr;
        const viewHeight = this.canvas.height / this.dpr;

        // Create transformation matrix
        const transform = new Float32Array([
            this.zoomScale, 0, this.offsetX,
            0, this.zoomScale, this.offsetY,
            0, 0, 1
        ]);

        this.webglRenderer.render(transform, {
            width: this.canvas.width,
            height: this.canvas.height
        });

        // Draw lasso path on top of WebGL content using 2D canvas
        if (this.lassoPath && this.lassoPath.length > 0) {
            this.drawLassoPath();
        }
    }
    
    highlightSearchResults(searchResults) {
        super.highlightSearchResults(searchResults);

        if (this.useWebGL && this.webglRenderer) {
            this.webglRenderer.updateHighlighting(
                this.highlightedPoint ? [this.highlightedPoint] : [],
                searchResults
            );
        }
    }

    highlightLassoSelection(selectedIndices) {
        if (this.useWebGL && this.webglRenderer) {
            const selectedArray = Array.from(selectedIndices);
            this.webglRenderer.updateHighlighting(selectedArray, []);
        }
        this.requestRender();
    }

    enableMetadataFilterMode(filteredIndices) {
        super.enableMetadataFilterMode(filteredIndices);
        if (this.useWebGL && this.webglRenderer) {
            this.webglRenderer.applyMetadataFilter(filteredIndices);
        }
    }

    disableMetadataFilterMode() {
        super.disableMetadataFilterMode();
        if (this.useWebGL && this.webglRenderer) {
            this.webglRenderer.applyMetadataFilter(null);
        }
    }

    // Keep GL buffer sizes aligned with canvas pixel size
    resizeCanvas() {
        super.resizeCanvas();
        if (this.useWebGL) {
            this.webglRenderer.resize(this.canvas.width, this.canvas.height);
        }
    }
    
    destroy() {
        super.destroy();
        
        if (this.webglRenderer) {
            this.webglRenderer.destroy();
            this.webglRenderer = null;
        }
    }
}

// Make enhanced visualization available globally
window.EnhancedCanvasVisualization = EnhancedCanvasVisualization;
window.WebGLRenderer = WebGLRenderer;
