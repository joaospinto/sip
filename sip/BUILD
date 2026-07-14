load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "sip",
    srcs = [
        "helpers.cpp",
        "sip.cpp",
        "types.cpp",
    ],
    hdrs = [
        "helpers.hpp",
        "sip.hpp",
        "types.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = ["@fmt"],
)
