target("llaisys-device-musa")
    set_kind("static")
    set_languages("cxx17")
    set_warnings("all", "error")

    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_includedirs("/usr/local/musa/include")
    add_linkdirs("/usr/local/musa/lib", {public = true})
    add_links("musart", {public = true})

    add_files("../src/device/musa/*.cpp")

    on_install(function (target) end)
target_end()
