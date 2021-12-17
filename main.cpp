#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <optional>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// TODO I get the following errors from the validation layers:
// validation layer: /usr/lib/i386-linux-gnu/libvulkan_radeon.so: wrong ELF class: ELFCLASS32
// validation layer: /usr/lib/i386-linux-gnu/libvulkan_intel.so: wrong ELF class: ELFCLASS32
// validation layer: /usr/lib/i386-linux-gnu/libvulkan_lvp.so: wrong ELF class: ELFCLASS32

/**
 * Wrapper around vkCreateDebugUtilsMessengerEXT for registering a debug callback
 * NOTE that pAllocator is assumed to be NULL
 */
VkResult CreateDebugUtilsMessengerEXT (VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, nullptr, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

/**
 * Wrapper around vkDestroyDebugUtilsMessengerEXT for destroying a debug callback
 * NOTE that pAllocator is assumed to be NULL
 */
void DestroyDebugUtilsMessengerEXT (VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, nullptr);
  }
}

struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;

  bool isComplete () {
    return graphicsFamily.has_value();
  }
};

/**
 * Find queue families for a device.
 */
QueueFamilyIndices findQueueFamilies (VkPhysicalDevice device) {
  QueueFamilyIndices indices;

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

  std::cout << "Number of queue families found: " << queueFamilies.size() << "\n";

  int i = 0;
  for (const auto& queueFamily : queueFamilies) {
    if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      indices.graphicsFamily = i;
      std::cout << "Found queue family with VK_QUEUE_GRAPHICS_BIT set\n";
      std::cout << "\tNumber of queues: " << queueFamily.queueCount << "\n";
      std::cout << "\tMinimum granularity width: " << queueFamily.minImageTransferGranularity.width << "\n";
      std::cout << "\tMinimum granularity height: " << queueFamily.minImageTransferGranularity.height << "\n";
      std::cout << "\tMinimum granularity depth: " << queueFamily.minImageTransferGranularity.depth << "\n";
    }

    if (indices.isComplete()) {
      break;
    }

    i++;
  }

  return indices;
}

bool isDeviceSuitable (VkPhysicalDevice device) {
  // Example code of picking a device based on some device properties and device features
  // This fails on my machine!
  // VkPhysicalDeviceProperties deviceProperties;
  // vkGetPhysicalDeviceProperties(device, &deviceProperties);

  // VkPhysicalDeviceFeatures deviceFeatures;
  // vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

  // return (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
  //         deviceFeatures.geometryShader);

  // NOTE there can be different ways to pick the device, the tutorial mentions
  // giving a score and sorting the devices according to that score and picking
  // the one with the highest score etc.

  QueueFamilyIndices indices = findQueueFamilies(device);
  return indices.graphicsFamily.has_value();
}

class HelloTriangleApplication {
public:
  void run () {
    initWindow();
    initVulkan();
    printAvaiableInstanceExtensions();
    testCode();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow* window;

  VkInstance instance;
  VkDebugUtilsMessengerEXT debugMessenger;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device;

  void initWindow () {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }

  void initVulkan () {
    createInstance();
    setupDebugMessenger();
    pickPhysicalDevice();
    createLogicalDevice();
  }

  void testCode () {
    //std::cout << std::boolalpha << "foo " << false << std::endl;
    //std::cout << std::noboolalpha << "bar " << true << std::endl;
    //std::cout << "foo2 " << false << std::endl;
  }

  void mainLoop () {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
    }
  }

  void cleanup () {
    if (enableValidationLayers) {
      // NOTE that if the following line is commented out, then the validation layers
      // will complain on the debug callback not being destroyed properly before the
      // instance is destroyed.
      DestroyDebugUtilsMessengerEXT(instance, debugMessenger);
    }

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
  }

  void createInstance () {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
      throw std::runtime_error("validation layers requested, but not available!");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers) {
      createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
      populateDebugMessengerCreateInfo(debugCreateInfo);
      createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    } else {
      createInfo.enabledLayerCount = 0;
      createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
      throw std::runtime_error("failed to create instance!");
    }
  }

  void populateDebugMessengerCreateInfo (VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
  }

  void setupDebugMessenger () {
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, &debugMessenger) != VK_SUCCESS) {
      throw std::runtime_error("failed to set up debug messenger!");
    }
  }

  /**
   * A physical device can have different properties. We pick the first one with
   * the VK_QUEUE_GRAPHICS_BIT set.
   */
  void pickPhysicalDevice () {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) {
      throw std::runtime_error("failed to find GPUs with Vulkan support");
    }

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    for (const auto& device : devices) {
      if (isDeviceSuitable(device)) {
        physicalDevice = device;
        break;
      }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
      throw std::runtime_error("failed to find suitable GPU");
    }
  }

  /**
   * A logical device interfaces with the physical device
   */
  void createLogicalDevice () {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
    queueCreateInfo.queueCount = 1;

    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    // Leaving this empty for now. Getting back to this later.
    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    // NOTE that we can provide an array of VkDeviceQueuecreateinfo here if we wanted to create
    // more than one queue family for this logical device.
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;
    // No device specific extensions are needed for now.
    createInfo.enabledExtensionCount = 0;

    if (enableValidationLayers) {
      // NOTE this is no longer necessary for modern vulkan implementations since validation layers
      // are no longer separate for instances and devices. (since when?).
      // This is only to be compatible with older implementations.
      createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
      createInfo.ppEnabledLayerNames = validationLayers.data();
    } else {
      createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
      throw std::runtime_error("failed to create logical device");
    }
  }

  std::vector<const char*> getRequiredExtensions () {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers) {
      extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    std::cout << "Required extensions (" << extensions.size() << "):\n";
    for (const auto& extension : extensions) {
      std::cout << '\t' <<  extension << "\n";
    }

    return extensions;
  }

  bool checkValidationLayerSupport () {
    uint32_t layerCount;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char* layerName : validationLayers) {
      bool layerFound = false;

      for (const auto& layerProperties : availableLayers) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }

      if (!layerFound) {
        return false;
      }
    }

    return true;
  }

  void printAvaiableInstanceExtensions () {
    uint32_t size = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &size, nullptr);
    std::vector<VkExtensionProperties> extensions(size);

    vkEnumerateInstanceExtensionProperties(nullptr, &size, extensions.data());
    std::cout << "Available instance extensions (" << extensions.size() << "):\n";
    for (const auto& extension : extensions) {
      std::cout << '\t' << extension.extensionName << '\n';
    }
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback (VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
  }
};

int main() {
  HelloTriangleApplication app;

  try {
    app.run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
