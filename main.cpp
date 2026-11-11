// 教学目标：看清 Vulkan 的“上传三角形 -> BLAS/TLAS -> Compute Ray Query -> 回读”。
// 只依赖 Vulkan SDK/C++17；无窗口、模型文件、纹理、数学库或引擎封装。
#include <vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation)+": VkResult="+std::to_string(result));
}
#define VK_CHECK(call) check((call), #call)

// 与 shaders/common.glsl 一致，所有 GPU 数据字段均为 16 字节。
struct V4 { float x=0, y=0, z=0, w=0; };
struct U4 { uint32_t x=0, y=0, z=0, w=0; };
static V4 operator+(V4 a, V4 b) { return {a.x+b.x,a.y+b.y,a.z+b.z,0}; }
static V4 operator*(V4 a, float s) { return {a.x*s,a.y*s,a.z*s,0}; }
static float length(V4 v) { return std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z); }
struct Face { V4 origin, edgeU, edgeV, albedo, emission; U4 grid; };
struct Surfel { V4 position; U4 info; };
struct Settings {
    uint32_t width, height, surfelCount, sampleCount, bounce;
    float indirectScale;
};
static_assert(sizeof(Face)==96 && sizeof(Surfel)==32 && sizeof(Settings)==24);

struct Scene {
    std::vector<V4> vertices;
    std::vector<Face> faces;
    std::vector<Surfel> surfels;

    void quad(V4 origin, V4 u, V4 v, V4 color, V4 emission={}) {
        // 位置与法线方向由 u x v 决定；每个矩形必须连续写两个三角形。
        uint32_t f=static_cast<uint32_t>(faces.size());
        uint32_t nx=std::max(1u,static_cast<uint32_t>(std::ceil(length(u)/0.10f)));
        uint32_t ny=std::max(1u,static_cast<uint32_t>(std::ceil(length(v)/0.10f)));
        faces.push_back({origin,u,v,color,emission,
            {static_cast<uint32_t>(surfels.size()),nx,ny,0}});
        for(uint32_t y=0;y<ny;++y)
            for(uint32_t x=0;x<nx;++x)
                surfels.push_back({origin+u*((x+0.5f)/nx)+v*((y+0.5f)/ny),{f,0,0,0}});
        for(V4 p : {origin,origin+u,origin+u+v,origin,origin+u+v,origin+v})
            vertices.push_back(p);
    }
    void box(float x, float z, float width, float depth, float height) {
        const V4 white{0.72f,0.72f,0.72f,0};
        quad({x,0,z+depth},{width,0,0},{0,height,0},white); // 前，法线 +z
        quad({x+width,0,z},{-width,0,0},{0,height,0},white); // 后，法线 -z
        quad({x,0,z},{0,0,depth},{0,height,0},white); // 左，法线 -x
        quad({x+width,0,z+depth},{0,0,-depth},{0,height,0},white); // 右，法线 +x
        quad({x,height,z+depth},{width,0,0},{0,0,-depth},white); // 顶，法线 +y
    }
    Scene() {
        const V4 white{0.72f,0.72f,0.72f,0};
        quad({-1,0,1},{2,0,0},{0,0,-2},white); // 0 地板
        quad({-1,2,-1},{2,0,0},{0,0,2},white); // 1 天花板
        quad({-1,0,-1},{2,0,0},{0,2,0},white); // 2 后墙
        quad({-1,0,1},{0,0,-2},{0,2,0},{0.75f,0.08f,0.06f,0}); // 3 红墙
        quad({1,0,-1},{0,0,2},{0,2,0},{0.08f,0.62f,0.12f,0}); // 4 绿墙
        quad({-0.35f,1.99f,-0.35f},{0.7f,0,0},{0,0,0.7f},{},
             {12,11,9,0}); // 5 面积灯，向下发光；GLSL 用 LIGHT_FACE=5 查找
        box(-0.65f,-0.05f,0.55f,0.60f,0.55f);
        box(0.15f,-0.65f,0.55f,0.60f,1.10f);
    }
};

struct Buffer {
    VkBuffer handle=VK_NULL_HANDLE;
    VkDeviceMemory memory=VK_NULL_HANDLE;
    VkDeviceSize size=0;
    VkDeviceAddress address=0; // 只有请求 SHADER_DEVICE_ADDRESS 的缓冲区才有地址
};

// 资源统一归此对象所有；析构按依赖的逆序释放，异常路径也不会漏 Vulkan 对象。
struct Vulkan {
    VkInstance instance=VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger=VK_NULL_HANDLE;
    VkPhysicalDevice gpu=VK_NULL_HANDLE;
    VkDevice device=VK_NULL_HANDLE;
    VkQueue queue=VK_NULL_HANDLE;
    VkCommandPool commandPool=VK_NULL_HANDLE;
    VkCommandBuffer commands=VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout=VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool=VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout=VK_NULL_HANDLE;
    uint32_t family=0, validationErrors=0;
    VkDeviceSize scratchAlignment=256;
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    std::vector<Buffer> buffers;
    std::vector<VkAccelerationStructureKHR> structures;
    std::vector<VkPipeline> pipelines;
    PFN_vkCreateAccelerationStructureKHR createAS=nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroyAS=nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes=nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAS=nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getASAddress=nullptr;

    static VKAPI_ATTR VkBool32 VKAPI_CALL debug(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
        if(severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
            ++static_cast<Vulkan*>(user)->validationErrors;
        std::cerr << "[validation] " << data->pMessage << '\n';
        return VK_FALSE;
    }

    void init(bool validate) {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName="Surfel Vulkan Minimal";
        app.apiVersion=VK_API_VERSION_1_2;
        const char* layer="VK_LAYER_KHRONOS_validation";
        const char* instanceExtensions[]={VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
                                         VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME};
        VkDebugUtilsMessengerCreateInfoEXT debugInfo{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        debugInfo.messageSeverity=VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                 |VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugInfo.messageType=VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                             |VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                             |VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugInfo.pfnUserCallback=debug; debugInfo.pUserData=this;
        VkValidationFeatureEnableEXT sync=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
        VkValidationFeaturesEXT validation{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
        validation.enabledValidationFeatureCount=1; validation.pEnabledValidationFeatures=&sync;
        validation.pNext=&debugInfo;
        VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        info.pApplicationInfo=&app;
        if(validate) {
            info.enabledLayerCount=1; info.ppEnabledLayerNames=&layer;
            info.enabledExtensionCount=2; info.ppEnabledExtensionNames=instanceExtensions;
            info.pNext=&validation;
        }
        VK_CHECK(vkCreateInstance(&info,nullptr,&instance));
        if(validate) {
            auto create=reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance,"vkCreateDebugUtilsMessengerEXT"));
            if(!create) throw std::runtime_error("Debug utils unavailable");
            VK_CHECK(create(instance,&debugInfo,nullptr,&messenger));
        }

        // 既检查扩展，也检查 feature 位；有 Vulkan 驱动不代表支持硬件 Ray Query。
        uint32_t count=0; VK_CHECK(vkEnumeratePhysicalDevices(instance,&count,nullptr));
        std::vector<VkPhysicalDevice> devices(count);
        VK_CHECK(vkEnumeratePhysicalDevices(instance,&count,devices.data()));
        const std::array<const char*,3> extensions={VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            VK_KHR_RAY_QUERY_EXTENSION_NAME,VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
        for(auto candidate:devices) {
            VkPhysicalDeviceProperties properties; vkGetPhysicalDeviceProperties(candidate,&properties);
            if(properties.apiVersion<VK_API_VERSION_1_2) continue;
            uint32_t n=0; VK_CHECK(vkEnumerateDeviceExtensionProperties(candidate,nullptr,&n,nullptr));
            std::vector<VkExtensionProperties> available(n);
            VK_CHECK(vkEnumerateDeviceExtensionProperties(candidate,nullptr,&n,available.data()));
            bool supported=true;
            for(auto name:extensions) {
                supported &= std::any_of(available.begin(),available.end(),[&](auto& e){
                    return std::strcmp(e.extensionName,name)==0;
                });
            }
            if(!supported) continue;
            VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
            VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
            VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext=&v12; v12.pNext=&as; as.pNext=&rq;
            vkGetPhysicalDeviceFeatures2(candidate,&features);
            if(!rq.rayQuery || !as.accelerationStructure || !v12.bufferDeviceAddress) continue;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,nullptr);
            std::vector<VkQueueFamilyProperties> families(n);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate,&n,families.data());
            for(uint32_t i=0;i<n;++i) if(families[i].queueFlags&VK_QUEUE_COMPUTE_BIT) {
                gpu=candidate; family=i; break;
            }
            if(gpu) { std::cout << "GPU: " << properties.deviceName << '\n'; break; }
        }
        if(!gpu) throw std::runtime_error("No Vulkan 1.2 GPU with accelerationStructure + rayQuery + bufferDeviceAddress");
        vkGetPhysicalDeviceMemoryProperties(gpu,&memoryProperties);
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext=&asProps; vkGetPhysicalDeviceProperties2(gpu,&properties);
        scratchAlignment=asProps.minAccelerationStructureScratchOffsetAlignment;

        float priority=1;
        VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex=family; q.queueCount=1; q.pQueuePriorities=&priority;
        VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
        rq.rayQuery=VK_TRUE;
        VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
        as.accelerationStructure=VK_TRUE; as.pNext=&rq;
        VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        v12.bufferDeviceAddress=VK_TRUE; v12.pNext=&as;
        VkDeviceCreateInfo d{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        d.pNext=&v12; d.queueCreateInfoCount=1; d.pQueueCreateInfos=&q;
        d.enabledExtensionCount=static_cast<uint32_t>(extensions.size()); d.ppEnabledExtensionNames=extensions.data();
        VK_CHECK(vkCreateDevice(gpu,&d,nullptr,&device));
        vkGetDeviceQueue(device,family,0,&queue);

        // KHR 扩展函数通过设备查询；Vulkan 1.2 核心函数可直接调用。
#define LOAD(member, name) member=reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device,#name)); \
        if(!member) throw std::runtime_error(#name " unavailable")
        LOAD(createAS,vkCreateAccelerationStructureKHR);
        LOAD(destroyAS,vkDestroyAccelerationStructureKHR);
        LOAD(getBuildSizes,vkGetAccelerationStructureBuildSizesKHR);
        LOAD(cmdBuildAS,vkCmdBuildAccelerationStructuresKHR);
        LOAD(getASAddress,vkGetAccelerationStructureDeviceAddressKHR);
#undef LOAD
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.queueFamilyIndex=family;
        VK_CHECK(vkCreateCommandPool(device,&pool,nullptr,&commandPool));
        VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        alloc.commandPool=commandPool; alloc.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; alloc.commandBufferCount=1;
        VK_CHECK(vkAllocateCommandBuffers(device,&alloc,&commands));
    }

    Buffer buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool hostVisible, const void* initial=nullptr) {
        Buffer result; result.size=size;
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size=size; info.usage=usage; info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device,&info,nullptr,&result.handle));
        // 立即登记所有权，即使后续分配失败也会释放已创建的 buffer。
        buffers.push_back(result); Buffer& owned=buffers.back();
        VkMemoryRequirements req; vkGetBufferMemoryRequirements(device,result.handle,&req);
        VkMemoryPropertyFlags flags=hostVisible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                               : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        uint32_t type=UINT32_MAX;
        for(uint32_t i=0;i<memoryProperties.memoryTypeCount;++i)
            if((req.memoryTypeBits&(1u<<i)) && (memoryProperties.memoryTypes[i].propertyFlags&flags)==flags) {
                type=i; break;
            }
        if(type==UINT32_MAX) throw std::runtime_error("Required memory type unavailable");
        VkMemoryAllocateFlagsInfo addressFlag{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        addressFlag.flags=VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize=req.size; alloc.memoryTypeIndex=type;
        if(usage&VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) alloc.pNext=&addressFlag;
        VK_CHECK(vkAllocateMemory(device,&alloc,nullptr,&owned.memory));
        VK_CHECK(vkBindBufferMemory(device,owned.handle,owned.memory,0));
        if(hostVisible) {
            void* mapped; VK_CHECK(vkMapMemory(device,owned.memory,0,size,0,&mapped));
            if(initial) std::memcpy(mapped,initial,static_cast<size_t>(size));
            else std::memset(mapped,0,static_cast<size_t>(size));
            vkUnmapMemory(device,owned.memory);
            // HOST_COHERENT 省去 flush；提交前的 host 写入对设备可见。
        }
        if(usage&VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
            VkBufferDeviceAddressInfo address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            address.buffer=owned.handle; owned.address=vkGetBufferDeviceAddress(device,&address);
        }
        return owned;
    }

    void begin() {
        VK_CHECK(vkResetCommandPool(device,commandPool,0));
        VkCommandBufferBeginInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        info.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(commands,&info));
    }
    void submit() {
        VK_CHECK(vkEndCommandBuffer(commands));
        VkSubmitInfo info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        info.commandBufferCount=1; info.pCommandBuffers=&commands;
        VK_CHECK(vkQueueSubmit(queue,1,&info,VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(queue)); // 教学版只做离线渲染；实时程序应使用 fence/多帧资源
    }
    void barrier(VkPipelineStageFlags sourceStage, VkAccessFlags sourceAccess,
                 VkPipelineStageFlags targetStage, VkAccessFlags targetAccess) {
        VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask=sourceAccess; b.dstAccessMask=targetAccess;
        vkCmdPipelineBarrier(commands,sourceStage,targetStage,0,1,&b,0,nullptr,0,nullptr);
    }

    VkAccelerationStructureKHR buildAS(VkAccelerationStructureTypeKHR type,
        const VkAccelerationStructureGeometryKHR& geometry, uint32_t primitiveCount) {
        VkAccelerationStructureBuildGeometryInfoKHR build{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        build.type=type; build.flags=VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        build.mode=VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        build.geometryCount=1; build.pGeometries=&geometry;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        getBuildSizes(device,VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,&build,&primitiveCount,&sizes);
        Buffer storage=buffer(sizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,false);
        VkAccelerationStructureCreateInfoKHR create{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        create.type=type; create.buffer=storage.handle; create.size=sizes.accelerationStructureSize;
        VkAccelerationStructureKHR structure;
        VK_CHECK(createAS(device,&create,nullptr,&structure)); structures.push_back(structure);
        // scratch 是构建期间的临时空间，设备地址必须满足专门的对齐要求。
        Buffer scratch=buffer(sizes.buildScratchSize+scratchAlignment,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,false);
        build.dstAccelerationStructure=structure;
        build.scratchData.deviceAddress=(scratch.address+scratchAlignment-1)&~(scratchAlignment-1);
        VkAccelerationStructureBuildRangeInfoKHR range{}; range.primitiveCount=primitiveCount;
        const auto* rangePointer=&range;
        begin(); cmdBuildAS(commands,1,&build,&rangePointer);
        barrier(VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
        submit();
        return structure;
    }

    VkPipeline pipeline(const char* name) {
        std::filesystem::path path=std::filesystem::path(SHADER_DIR)/(std::string(name)+".spv");
        std::ifstream input(path,std::ios::binary|std::ios::ate);
        if(!input) throw std::runtime_error("Cannot open shader: "+path.string());
        size_t size=static_cast<size_t>(input.tellg());
        std::vector<uint32_t> code(size/4); input.seekg(0);
        if(!input.read(reinterpret_cast<char*>(code.data()),static_cast<std::streamsize>(size)))
            throw std::runtime_error("Cannot read SPIR-V");
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize=size; moduleInfo.pCode=code.data();
        VkShaderModule module; VK_CHECK(vkCreateShaderModule(device,&moduleInfo,nullptr,&module));
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT; info.stage.module=module; info.stage.pName="main";
        info.layout=pipelineLayout;
        VkPipeline result=VK_NULL_HANDLE;
        VkResult status=vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&info,nullptr,&result);
        vkDestroyShaderModule(device,module,nullptr);
        VK_CHECK(status); pipelines.push_back(result); return result;
    }

    ~Vulkan() {
        if(device) {
            vkDeviceWaitIdle(device);
            for(auto p:pipelines) vkDestroyPipeline(device,p,nullptr);
            if(pipelineLayout) vkDestroyPipelineLayout(device,pipelineLayout,nullptr);
            if(descriptorPool) vkDestroyDescriptorPool(device,descriptorPool,nullptr);
            if(setLayout) vkDestroyDescriptorSetLayout(device,setLayout,nullptr);
            for(auto as:structures) destroyAS(device,as,nullptr);
            for(auto b:buffers) {
                vkDestroyBuffer(device,b.handle,nullptr);
                if(b.memory) vkFreeMemory(device,b.memory,nullptr);
            }
            if(commandPool) vkDestroyCommandPool(device,commandPool,nullptr);
            vkDestroyDevice(device,nullptr);
        }
        if(messenger) {
            auto destroy=reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance,"vkDestroyDebugUtilsMessengerEXT"));
            destroy(instance,messenger,nullptr);
        }
        if(instance) vkDestroyInstance(instance,nullptr);
    }
};

// 手写最简单的 32-bit BMP，避免引入图像库；负高度表示自顶向下保存。
static void saveBMP(const char* name, Vulkan& vk, Buffer pixels, uint32_t width, uint32_t height) {
    std::ofstream out(name,std::ios::binary);
    auto u16=[&](uint16_t x){out.write(reinterpret_cast<const char*>(&x),2);};
    auto u32=[&](uint32_t x){out.write(reinterpret_cast<const char*>(&x),4);};
    u16(0x4d42); u32(54+width*height*4); u32(0); u32(54);
    u32(40); u32(width); u32(static_cast<uint32_t>(-static_cast<int32_t>(height)));
    u16(1); u16(32); u32(0); u32(width*height*4); u32(2835); u32(2835); u32(0); u32(0);
    void* mapped; VK_CHECK(vkMapMemory(vk.device,pixels.memory,0,pixels.size,0,&mapped));
    out.write(static_cast<const char*>(mapped),static_cast<std::streamsize>(pixels.size));
    vkUnmapMemory(vk.device,pixels.memory);
    if(!out) throw std::runtime_error(std::string("Cannot write ")+name);
    std::cout << "Saved " << name << '\n';
}

int main(int argc, char** argv) {
    try {
        bool validate=false;
        for(int i=1;i<argc;++i) {
            if(std::string(argv[i])=="--validate") validate=true;
            else throw std::runtime_error("Usage: surfel_vulkan [--validate]");
        }
        // 学习时先调整这几个常量：缓存密度在 Scene::quad 中，默认间距为 0.1。
        constexpr uint32_t Width=640, Height=640, Samples=256, Bounces=3;
        Scene scene;
        Vulkan vk; vk.init(validate);
        std::cout << "Triangles: " << scene.vertices.size()/3 << ", surfels: " << scene.surfels.size()
                  << ", samples/surfel: " << Samples << ", indirect bounces: " << Bounces << '\n';
        const VkBufferUsageFlags asInput=VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                       |VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        Buffer vertices=vk.buffer(scene.vertices.size()*sizeof(V4),asInput,true,scene.vertices.data());
        Buffer faces=vk.buffer(scene.faces.size()*sizeof(Face),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,true,scene.faces.data());
        Buffer surfels=vk.buffer(scene.surfels.size()*sizeof(Surfel),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,true,scene.surfels.data());
        std::array<Buffer,2> cache={
            vk.buffer(scene.surfels.size()*sizeof(V4),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,true),
            vk.buffer(scene.surfels.size()*sizeof(V4),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,true)};
        Buffer pixels=vk.buffer(Width*Height*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,true);

        // BLAS：描述真实三角形。顶点步长为 16 字节，但格式只读取 xyz 三个 float。
        VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometry.geometryType=VK_GEOMETRY_TYPE_TRIANGLES_KHR; geometry.flags=VK_GEOMETRY_OPAQUE_BIT_KHR;
        auto& triangles=geometry.geometry.triangles;
        triangles.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.vertexFormat=VK_FORMAT_R32G32B32_SFLOAT; triangles.vertexData.deviceAddress=vertices.address;
        triangles.vertexStride=sizeof(V4); triangles.maxVertex=static_cast<uint32_t>(scene.vertices.size()-1);
        triangles.indexType=VK_INDEX_TYPE_NONE_KHR;
        auto blas=vk.buildAS(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,geometry,
                             static_cast<uint32_t>(scene.vertices.size()/3));

        // TLAS：本例只引用一个 BLAS 实例。实例矩阵用于物体变换，不必改三角形数据。
        VkAccelerationStructureDeviceAddressInfoKHR address{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        address.accelerationStructure=blas;
        VkAccelerationStructureInstanceKHR instance{};
        instance.transform.matrix[0][0]=instance.transform.matrix[1][1]=instance.transform.matrix[2][2]=1;
        instance.mask=0xff;
        instance.flags=VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference=vk.getASAddress(vk.device,&address);
        Buffer instances=vk.buffer(sizeof(instance),asInput,true,&instance);
        VkAccelerationStructureGeometryKHR top{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        top.geometryType=VK_GEOMETRY_TYPE_INSTANCES_KHR;
        top.geometry.instances.sType=VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        top.geometry.instances.data.deviceAddress=instances.address;
        auto tlas=vk.buildAS(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,top,1);

        // Descriptor 是着色器的“资源插座”，binding 编号对应 common.glsl。
        std::array<VkDescriptorSetLayoutBinding,6> bindings{};
        for(uint32_t i=0;i<bindings.size();++i) {
            bindings[i].binding=i; bindings[i].descriptorCount=1;
            bindings[i].descriptorType=i==0 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout.bindingCount=static_cast<uint32_t>(bindings.size()); layout.pBindings=bindings.data();
        VK_CHECK(vkCreateDescriptorSetLayout(vk.device,&layout,nullptr,&vk.setLayout));
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(Settings)};
        VkPipelineLayoutCreateInfo pipeLayout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeLayout.setLayoutCount=1; pipeLayout.pSetLayouts=&vk.setLayout;
        pipeLayout.pushConstantRangeCount=1; pipeLayout.pPushConstantRanges=&push;
        VK_CHECK(vkCreatePipelineLayout(vk.device,&pipeLayout,nullptr,&vk.pipelineLayout));
        VkDescriptorPoolSize poolSizes[]={{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,2},
                                         {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,10}};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets=2; pool.poolSizeCount=2; pool.pPoolSizes=poolSizes;
        VK_CHECK(vkCreateDescriptorPool(vk.device,&pool,nullptr,&vk.descriptorPool));
        VkDescriptorSetLayout layouts[]={vk.setLayout,vk.setLayout};
        VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        alloc.descriptorPool=vk.descriptorPool; alloc.descriptorSetCount=2; alloc.pSetLayouts=layouts;
        std::array<VkDescriptorSet,2> sets{};
        VK_CHECK(vkAllocateDescriptorSets(vk.device,&alloc,sets.data()));
        for(uint32_t i=0;i<2;++i) {
            // set[0]: 读 cache[0]、写 cache[1]；set[1]: 反过来，称为 ping-pong。
            Buffer resources[]={faces,surfels,cache[i],cache[1-i],pixels};
            std::array<VkDescriptorBufferInfo,5> bufferInfos{};
            std::array<VkWriteDescriptorSet,6> writes{};
            VkWriteDescriptorSetAccelerationStructureKHR asInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            asInfo.accelerationStructureCount=1; asInfo.pAccelerationStructures=&tlas;
            for(uint32_t b=0;b<6;++b) {
                writes[b].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet=sets[i]; writes[b].dstBinding=b; writes[b].descriptorCount=1;
                writes[b].descriptorType=bindings[b].descriptorType;
                if(b==0) writes[b].pNext=&asInfo;
                else {
                    bufferInfos[b-1]={resources[b-1].handle,0,resources[b-1].size};
                    writes[b].pBufferInfo=&bufferInfos[b-1];
                }
            }
            vkUpdateDescriptorSets(vk.device,6,writes.data(),0,nullptr);
        }
        auto integrate=vk.pipeline("integrate"), resolve=vk.pipeline("resolve");
        Settings settings{Width,Height,static_cast<uint32_t>(scene.surfels.size()),Samples,0,1};
        auto bind=[&](VkPipeline pipeline, VkDescriptorSet set) {
            vkCmdBindPipeline(vk.commands,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
            vkCmdBindDescriptorSets(vk.commands,VK_PIPELINE_BIND_POINT_COMPUTE,vk.pipelineLayout,0,1,&set,0,nullptr);
            vkCmdPushConstants(vk.commands,vk.pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(settings),&settings);
        };

        auto start=std::chrono::steady_clock::now();
        vk.begin();
        for(uint32_t bounce=0;bounce<Bounces;++bounce) {
            settings.bounce=bounce;
            bind(integrate,sets[bounce%2]);
            // 这就是一个计算阶段：发出工作组，每组 64 个线程，每线程一个 surfel。
            vkCmdDispatch(vk.commands,(settings.surfelCount+63)/64,1,1);
            // 光有 dispatch 顺序还不够：明确让上一轮写入对下一轮读取可见。
            vk.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT);
        }
        vk.submit();
        for(uint32_t withGI=0;withGI<2;++withGI) {
            vk.begin(); settings.indirectScale=static_cast<float>(withGI);
            bind(resolve,sets[Bounces%2]); // 此 set 的 Previous binding 指向最后一轮输出
            vkCmdDispatch(vk.commands,(Width+7)/8,(Height+7)/8,1);
            vk.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_HOST_READ_BIT);
            vk.submit();
            saveBMP(withGI ? "gi.bmp" : "direct.bmp",vk,pixels,Width,Height);
        }
        double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        std::cout << "Integrate + 2 resolves + readback: " << ms << " ms (CPU wall time, not FPS)\n";
        std::cout << "Validation errors: " << vk.validationErrors << '\n';
        return vk.validationErrors ? 2 : 0;
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n'; return 1;
    }
}
