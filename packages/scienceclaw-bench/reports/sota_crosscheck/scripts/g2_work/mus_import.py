import sys, types
for n in ('stempeg','musdb','musdb.audio_classes'):
    sys.modules[n]=types.ModuleType(n)
sys.modules['musdb'].__path__=[]
try:
    import museval
except Exception as e:
    print('import museval failed', repr(e))
from museval import metrics
print(metrics.__file__)
import inspect
print(inspect.signature(metrics.bss_eval))
