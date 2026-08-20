// Shared Vulkan backend plumbing, and the Vulkan algorithm registry.
//
// This file exists because the Vulkan build was written assuming exactly one
// algorithm. miner_vk.cpp hardcoded `if (want != "autolykos2")`, --list-algos
// printed a literal string, and the device name/VRAM/driver readout was read
// out of a `g_instance` pointer that only Autolykos ever set. Adding a second
// algorithm without this would have meant two copies of the device-selection
// logic and a symbol clash on the three readout functions.
//
// The split is: everything that is about the *device* lives here and is
// written once; everything that is about the *algorithm* lives in the
// algorithm's own algo_vk.cpp. The registry mirrors src/core/registry.cu, and
// for the same reason - an explicit table rather than static-initialiser
// self-registration, because with separate compilation a dropped object turns
// self-registration into a miner that quietly supports nothing.
//
// Adding a third Vulkan algorithm is now: a .comp, an algo_vk.cpp, and two
// lines in kVulkanRegistry.

#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

#include "algo.h"

namespace om {

/** Returns the Vulkan implementation registered under `name`, or nullptr.
 *  `deviceIndex` is -1 for "pick the largest card". */
Algorithm *createVulkanAlgorithm(const std::string &name, int deviceIndex);

/** Names of every algorithm this Vulkan build can actually run. */
std::vector<std::string> availableVulkanAlgorithms();

/**
 * Selects the physical device to run on: real GPUs only, largest device-local
 * heap wins unless an index is given.
 *
 * Shared rather than copied because the "only real GPUs" filter is load
 * bearing - llvmpipe/lavapipe advertise themselves as Vulkan devices of type
 * CPU and would "work", very slowly, while looking like a successful start.
 */
bool vkPickPhysicalDevice(VkInstance inst, int requestedIndex,
                          VkPhysicalDevice *out);

/**
 * Does this device really support int8 cooperative matrix, and at what shape?
 *
 * CHECK ORDER IS LOAD BEARING, and the obvious order is wrong.
 * vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR returns 14 configurations
 * on an RX 6700 XT, several of them int8, and that device supports cooperative
 * matrix not at all: VK_KHR_cooperative_matrix is absent from its extension
 * list, the feature bit is false, and vkCreateDevice fails with
 * VK_ERROR_FEATURE_NOT_PRESENT. The loader resolves the enumeration entry
 * point because another device on the same box has the extension, and then
 * answers for a device that does not.
 *
 * Measured 2026-08-20 on a box with three Vulkan devices, where the extension
 * and the feature bit both belonged to llvmpipe and neither belonged to the
 * 6700 XT.
 *
 * So: extension list first, then the feature bit, and only then the
 * configuration list. Returns false unless all three agree, and fills *kOut
 * with the K of the first sint8 x sint8 -> sint32 configuration when they do.
 * K differs by vendor - 32 on Ada, 16 on RDNA3 - which is why it is returned
 * rather than assumed, and why the shader takes it as a specialisation
 * constant.
 */
bool vkInt8CooperativeMatrix(VkInstance inst, VkPhysicalDevice dev,
                             uint32_t *kOut);

/**
 * Each backend reports the device it opened here, so the readout belongs to
 * the backend rather than to whichever algorithm happens to be compiled in.
 */
void vkSetDeviceInfo(const char *name, double memGB, const char *driver);

const char *vkDeviceName();
double vkDeviceMemGB();
const char *vkDriverVersion();

/** Prints every Vulkan GPU, for --list-devices. */
void vkListDevices();

const char *driverTypeName(int t);

}  // namespace om
