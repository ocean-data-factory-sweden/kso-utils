# base imports
import logging
import multiprocessing

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def import_model_modules(module_names):
    importlib = __import__("importlib")
    modules = {}
    for module_name, module_full in zip(["train", "detect", "val"], module_names):
        try:
            modules[module_name] = importlib.import_module(module_full)
        except ModuleNotFoundError:
            logging.error(f"Module {module_name} could not be imported.")
    return modules


def import_modules(module_names, utils: bool = True, models: bool = False):
    importlib = __import__("importlib")
    modules = {}
    model_presets = ["train", "detect", "val"]
    for i, module_name in enumerate(module_names):
        if utils:
            module_full = "kso_utils." + module_name
        else:
            module_full = module_name
        try:
            if models:
                module_name = model_presets[i]
            modules[module_name] = importlib.import_module(module_full)
        except ModuleNotFoundError:
            logging.error(f"Module {module_name} could not be imported.")
    return modules


def parallel_map(func, iterable, args=()):
    with multiprocessing.Pool() as pool:
        results = pool.starmap(func, zip(iterable, *args))
    return results
