# RFFickle: Roboflow Fork of Fickle

This is a fork of the [Fickle](https://pypi.org/project/fickle/) package by Eduard Christian Dumitrescu, with additional functionality added by Roboflow.

## Fork Information

- **Original Package**: [fickle](https://pypi.org/project/fickle/) v0.2.2
- **Original Author**: Eduard Christian Dumitrescu
- **Fork Maintainer**: Roboflow, Inc.
- **PyPI Package**: `rffickle`

This fork was created from the PyPI source distribution as the original source code was not available on GitHub.

---

# Original README: Fickle - Firewalled Pickle

People abuse pickle. Especially researchers. Pickle is [not secure](https://docs.python.org/3/library/pickle.html). Published datasets and ML training weights are often distributed as pickle files (or formats which use pickle files, such as [PyTorch checkpoint.ckpt files](https://pytorch.org/tutorials/beginner/saving_loading_models.html)). Sometimes it is the only format that they are available in.

## Examples

Loading basic types is easy:

```python
>>> from fickle import DefaultFirewall
>>> import pickle
>>>
>>> my_picked_data = pickle.dumps({"list": [1, 2, "three", b"four"]})
>>>
>>> firewall = DefaultFirewall()
>>> firewall.loads(my_picked_data)
{'list': [1, 2, 'three', b'four']}
```

Safely loading PyTorch checkpoint files into numpy arrays is just as easy:

```python
>>> from fickle.ext.pytorch import fake_torch_load_zipped
>>> from zipfile import ZipFile
>>>
>>> zf = ZipFile("/path/to/sd-v1-4.ckpt")
>>> ckpt = fake_torch_load_zipped(zf)
>>> tensor = ckpt["state_dict"]["model.diffusion_model.output_blocks.3.1.norm.weight"]
>>> tensor.array
array([0.39097363, 0.3898967 , 0.35191917, ..., 0.41924757, 0.4031702 ,
       0.37156993], dtype=float32)
```

You can, optionally, even use `marshmallow` for validation!

## Alternatives

- [picklemagic](https://github.com/CensoredUsername/picklemagic)
- [pikara](https://pypi.org/project/pikara)

|                                                | fickle | picklemagic | pikara |
|------------------------------------------------|--------|-------------|--------|
| Does not rely on `pickle._Unpickler`?          | ✅      | ❌           | ✅      |
| Uses `pickletools.genops`                      | yes    | no          | yes    |
| Can load without executing?                    | ✅      | ✅           | ?      |
| Forbid importing arbitrary objects?            | ✅      | ✅           | ?      |
| Forbid calling `list.append`/`set.add`/etc?    | ✅      | ❌           | ?      |
| Forbid calling all methods by default?         | ✅      | ❌           | ?      |
| Can create dangerous circular structures?      | ✅      | ✅           | ?      |
| Safe against billion laughs DoS attack?        | ?      | ?           | ?      |
| Full support for all pickle opcodes?           | ❌      | ✅           | ?      |
| Has unit tests?                                | ✅      | ❌           | ✅      |
| Stable API?                                    | ❌      | ✅           | ✅      |
