load("@rules_cc//cc:defs.bzl", "cc_library", "cc_test")

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

cc_test(
    name = "sip_test",
    srcs = ["sip_test.cpp"],
    linkstatic = True,
    deps = [":sip"],
)
