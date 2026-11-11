// CPU 与 GPU 的协议：全用 16 字节字段，避免 C++/std430 的 vec3 对齐陷阱。
// 一个 Face 是矩形表面：p = origin + u * edgeU + v * edgeV，u/v 在 [0,1]。
struct Face {
    vec4 origin;
    vec4 edgeU;
    vec4 edgeV;
    vec4 albedo;
    vec4 emission;
    uvec4 grid; // 首个 surfel 索引、横向数量、纵向数量、保留
};
struct Surfel {
    vec4 position;
    uvec4 info; // 所在 Face 的索引，其余保留
};
layout(set=0, binding=0) uniform accelerationStructureEXT scene;
layout(std430, set=0, binding=1) readonly buffer Faces { Face faces[]; };
layout(std430, set=0, binding=2) readonly buffer Surfels { Surfel surfels[]; };
layout(std430, set=0, binding=3) readonly buffer Previous { vec4 previousGI[]; };
layout(std430, set=0, binding=4) writeonly buffer Next { vec4 nextGI[]; };
layout(std430, set=0, binding=5) writeonly buffer Pixels { uint pixels[]; };
layout(push_constant) uniform Settings {
    uint width, height, surfelCount, sampleCount;
    uint bounce;
    float indirectScale; // Resolve 时：0=直接光，1=直接光+GI
} settings;

const float PI = 3.14159265359;
const float EPS = 0.001;
const uint LIGHT_FACE = 5u; // 与 main.cpp 的场景创建顺序保持一致

vec3 normalOf(uint f) {
    return normalize(cross(faces[f].edgeU.xyz, faces[f].edgeV.xyz));
}

uint randomBits(inout uint state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}
float randomFloat(inout uint state) {
    return float(randomBits(state) >> 8) * (1.0 / 16777216.0);
}

// 硬件光追的入口。没有自己遍历 BVH；驱动/GPU 负责遍历和三角形求交。
// 几何标为 opaque，因此不需要在循环里写 alpha-test 等候选命中处理。
bool trace(vec3 origin, vec3 direction, float maxT, out uint face, out vec3 point) {
    rayQueryEXT query;
    rayQueryInitializeEXT(query, scene, gl_RayFlagsOpaqueEXT, 0xff,
                          origin, 0.0, direction, maxT);
    while (rayQueryProceedEXT(query)) { }
    if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT)
        return false;
    // CPU 每个矩形连续写两个三角形，因此 primitiveIndex/2 就是 Face 索引。
    face = rayQueryGetIntersectionPrimitiveIndexEXT(query, true) / 2u;
    point = origin + direction * rayQueryGetIntersectionTEXT(query, true);
    return true;
}

// 遮挡只需要知道“有没有命中”，允许遇到任意遮挡物就提前结束。
bool blocked(vec3 origin, vec3 direction, float maxT) {
    rayQueryEXT query;
    rayQueryInitializeEXT(query, scene,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0xff, origin, 0.0, direction, maxT);
    while (rayQueryProceedEXT(query)) { }
    return rayQueryGetIntersectionTypeEXT(query, true) != gl_RayQueryCommittedIntersectionNoneEXT;
}

// 面积灯的显式采样，返回直接光的 E/pi（尚未乘当前表面的 albedo）。
// 在灯上均匀取样：pdf_area=1/area。几何项为 cosSurface*cosLight/distance²。
vec3 directDiffuse(vec3 point, vec3 normal, uint count, inout uint rng) {
    Face light = faces[LIGHT_FACE];
    vec3 result = vec3(0);
    float area = length(cross(light.edgeU.xyz, light.edgeV.xyz));
    for (uint i=0; i<count; ++i) {
        vec3 q = light.origin.xyz + randomFloat(rng)*light.edgeU.xyz
                                 + randomFloat(rng)*light.edgeV.xyz;
        // 偏移起点防止打到自身；距离与方向都从偏移后的起点算，保持一致。
        vec3 origin = point + normal*EPS;
        vec3 delta = q-origin;
        float distance2 = dot(delta, delta);
        float distance = sqrt(distance2);
        vec3 direction = delta/distance;
        float cosine = max(dot(normal, direction), 0.0)
                     * max(dot(normalOf(LIGHT_FACE), -direction), 0.0);
        if (cosine > 0.0 && !blocked(origin, direction, distance-EPS))
            result += light.emission.rgb * (cosine*area/(PI*distance2));
    }
    return result/float(count);
}

// 沿法线的余弦加权半球采样：pdf=cos(theta)/pi。
// 所以 (E/pi) 的蒙特卡洛估计正好是采样到的入射 radiance 的平均值。
vec3 cosineDirection(vec3 normal, inout uint rng) {
    float r = sqrt(randomFloat(rng));
    float phi = 2.0*PI*randomFloat(rng);
    vec3 tangent = normalize(cross(abs(normal.y)<0.99 ? vec3(0,1,0) : vec3(1,0,0), normal));
    vec3 bitangent = cross(normal, tangent);
    return tangent*(r*cos(phi)) + bitangent*(r*sin(phi)) + normal*sqrt(1.0-r*r);
}

// 本例的简化点：光追已经告诉我们命中了哪个矩形，可直接找到其 2x2 surfel。
// 在规则表面网格上双线性插值，不必实现原项目的三维哈希网格。
vec3 cachedGI(uint f, vec3 point) {
    Face face = faces[f];
    vec3 d = point-face.origin.xyz;
    vec2 uv = vec2(dot(d, face.edgeU.xyz)/dot(face.edgeU.xyz, face.edgeU.xyz),
                   dot(d, face.edgeV.xyz)/dot(face.edgeV.xyz, face.edgeV.xyz));
    ivec2 size = ivec2(face.grid.yz);
    vec2 coord = clamp(uv*vec2(size)-0.5, vec2(0), vec2(size-1));
    ivec2 a = ivec2(floor(coord));
    ivec2 b = min(a+1, size-1);
    vec2 t = fract(coord);
    uint start = face.grid.x;
    vec3 c00 = previousGI[start+uint(a.y*size.x+a.x)].rgb;
    vec3 c10 = previousGI[start+uint(a.y*size.x+b.x)].rgb;
    vec3 c01 = previousGI[start+uint(b.y*size.x+a.x)].rgb;
    vec3 c11 = previousGI[start+uint(b.y*size.x+b.x)].rgb;
    return mix(mix(c00,c10,t.x), mix(c01,c11,t.x), t.y);
}
