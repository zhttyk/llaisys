rule("musa.build")
    set_extensions(".mu")

    on_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local objectfile = target:objectfile(sourcefile)
        table.insert(target:objectfiles(), objectfile)

        local musa_root =
            os.getenv("MUSA_INSTALL_PATH") or "/usr/local/musa"

        local mcc =
            path.join(musa_root, "bin", "mcc")

        local include_dir =
            path.join(os.projectdir(), "include")

        batchcmds:show_progress(
            opt.progress,
            "${color.build.object}compiling.musa %s",
            sourcefile)

        batchcmds:mkdir(path.directory(objectfile))

        batchcmds:vrunv(mcc, {
            "-c",
            sourcefile,
            "-o",
            objectfile,
            "--offload-arch=mp_31",
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-I" .. include_dir
        })

        batchcmds:add_depfiles(sourcefile)
        batchcmds:set_depmtime(os.mtime(objectfile))
        batchcmds:set_depcache(target:dependfile(objectfile))
    end)
rule_end()


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


target("llaisys-ops-musa")
    set_kind("static")

    add_rules("musa.build")

    add_includedirs("/usr/local/musa/include")
    add_linkdirs("/usr/local/musa/lib", {public = true})
    add_links("musart", {public = true})

    add_files("../src/ops/*/musa/*.mu")

    on_install(function (target) end)
target_end()
