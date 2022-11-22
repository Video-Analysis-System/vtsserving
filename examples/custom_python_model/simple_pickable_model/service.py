import vtsserving

square_runner = vtsserving.picklable_model.get("my_python_model:latest").to_runner()

svc = vtsserving.Service("simple_square_svc", runners=[square_runner])


@svc.api(input=vtsserving.io.JSON(), output=vtsserving.io.JSON())
async def square(input_arr):
    return await square_runner.async_run(input_arr)
