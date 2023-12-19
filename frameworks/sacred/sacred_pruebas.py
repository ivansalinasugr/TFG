from sacred import Experiment

ex = Experiment()

@ex.config
def my_config():
    foo = 42
    bar = 'baz'

@ex.capture
def some_function(foo, bar):
    print('foo is ', foo)
    print('bar is ', bar)

@ex.automain
def my_main():
    some_function()
    for i in range(99911550):
    	None
