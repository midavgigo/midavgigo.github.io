var render_pipeline, compute_pipeline;
var canvas, context, presentationFormat;
var adapter, device;
var input, result, workBuffer, sideBuffer, timeBuffer, acclBuffer;
var computeBindGroup;
var renderBindGroup;
var device, adapter;
var renderPassDescriptor;

async function config_workspace(){
    adapter = await navigator.gpu?.requestAdapter();
    device = await adapter?.requestDevice();
    if (!device) {
        fail('need a browser that supports WebGPU');
        return false;
    }
    canvas = document.querySelector('canvas');
    context = canvas.getContext('webgpu');
    presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device,
        format: presentationFormat,
    });
    return true;
}

async function exec(){
    const compute_encoder = device.createCommandEncoder({
        label: 'compute_encoder'
    });
    const pass = compute_encoder.beginComputePass({
        label: 'compute pass'
    });
    pass.setPipeline(compute_pipeline);
    pass.setBindGroup(0, computeBindGroup);
    pass.dispatchWorkgroups(input.length);
    pass.end();

    const commandBuffer = compute_encoder.finish();
    device.queue.submit([commandBuffer]);
}

async function config_compute(){
    const module = this.device.createShaderModule({
        label: 'compute heights',
        code: `
            struct Param{
                height: f32,
                speed: f32
            }

            @group(0) @binding(0) var<storage, read_write>  data: array<Param>;
            @group(0) @binding(1) var<uniform>        side: u32;
            @group(0) @binding(2) var<uniform>        time: f32;
            @group(0) @binding(3) var<uniform>        accl: f32;

            @compute @workgroup_size(1) fn compute_height(@builtin(global_invocation_id) id: vec3u) {
                let size = side*side;
                let idx = id.x;
                let shifts = array(
                    -i32(side),
                    -1,
                    1,
                    i32(side)
                );
                var mid: f32 = 0;
                var count: f32 = 0;
                for(var i:u32; i < 4; i=i+1){
                    let temp_idx = i32(idx)+shifts[i];
                    if(temp_idx >= 0 && temp_idx < i32(size)){
                        mid = mid + data[temp_idx].height;
                        count = count + 1;
                    }
                }
                mid = mid / count;
                data[idx].speed = data[idx].speed/1.01 + (mid-data[idx].height)/2;
                data[idx].height += data[idx].speed - accl;
                
                data[0].height = 0;
                data[0].speed = 0;
                data[side-1].height = 0;
                data[side-1].speed = 0;
                data[side*side - side - 2].height = 0;
                data[side*side - side - 2].speed = 0;
                data[side*side - 1].height = 0;
                data[side*side - 1].speed = 0;

                data[side*side/2 - side/2].height = sin(time);
                data[side*side/2 - side/2].speed = 0;
            }
        `,
    });

    compute_pipeline = device.createComputePipeline({
        label: 'compute heights pipeline',
        layout: 'auto',
        compute: {
            module,
            entryPoint: 'compute_height',
        },
    });

    input = new Float32Array(512);
    input.fill(0, 0, 512);
    workBuffer = device.createBuffer({
        label: 'work buffer',
        size: input.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(workBuffer, 0, input);

    sideBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });
    const side = new Uint32Array([16]);
    device.queue.writeBuffer(sideBuffer, 0, side);

    timeBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    acclBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    computeBindGroup = device.createBindGroup({
        label: 'height bind group',
        layout: compute_pipeline.getBindGroupLayout(0),
        entries:[
            { binding: 0, resource:{ buffer: workBuffer}},
            { binding: 1, resource:{ buffer: sideBuffer}},
            { binding: 2, resource:{ buffer: timeBuffer}},
            { binding: 3, resource:{ buffer: acclBuffer}}
        ]
    });
}

async function render(){
    renderPassDescriptor.colorAttachments[0].view = context.getCurrentTexture().createView();
    const render_encoder = device.createCommandEncoder({label:"render encoder"});
    const pass = render_encoder.beginRenderPass(renderPassDescriptor);
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, renderBindGroup);
    pass.draw(450*3);
    pass.end();
    const commandBuffer = render_encoder.finish();
    device.queue.submit([commandBuffer]);
}

async function config_render(){
    const render_module = device.createShaderModule({
        label: 'render module',
        code: `
            struct VertexOut{
            @builtin(position) position: vec4f,
            @location(0) color: vec4f
        }

        struct Param{
            height: f32,
            speed: f32
        }

        @group(0) @binding(0) var<storage, read>  data: array<Param>;

        struct Cam{
            pos: vec3f,
            dir: vec3f
        }
        
        struct Screen{
            size: f32,
            indent: f32,
            top: vec3f
        }

        struct View{
            cam: Cam,
            screen: Screen
        }

        @group(0) @binding(1) var<storage, read> rawView: array<f32>;
        @group(0) @binding(2) var<uniform> side: u32;

        @vertex fn vertex_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOut{
            var view: View;
            view.cam.pos.x = rawView[0];
            view.cam.pos.y = rawView[1];
            view.cam.pos.z = rawView[2];
            view.cam.dir.x = rawView[3];
            view.cam.dir.y = rawView[4];
            view.cam.dir.z = rawView[5];
            view.screen.size = rawView[6];
            view.screen.indent = rawView[7];
            view.screen.top.x = rawView[8];
            view.screen.top.y = rawView[9];
            view.screen.top.z = rawView[10];
            var ret: VertexOut;
            var x = (vertexIndex/6) % (side - 1);
            var y = (vertexIndex/6) / (side - 1);
            var ind = vertexIndex % 6;
            var pos = array(
                vec2u(0, 0),
                vec2u(1, 0),
                vec2u(0, 1),
                vec2u(1, 0),
                vec2u(0, 1),
                vec2u(1, 1),
            );
            var colors = array(
                vec4f(0.5, 0.5, 0.5, 1.0),
                vec4f(0.5, 0.5, 1.0, 1.0),
                vec4f(0.5, 1.0, 0.5, 1.0),
                vec4f(0.5, 0.5, 1.0, 1.0),
                vec4f(0.5, 1.0, 0.5, 1.0),
                vec4f(1.0, 1.0, 1.0, 1.0),
            );
            x += pos[ind].x;
            y += pos[ind].y;
            var z = data[y*16+x].height;
            ret.color = colors[ind];

            var vc = normalize(vec3f(f32(x), f32(y), f32(z))-view.cam.pos);
            var d = -view.cam.dir*view.screen.indent;
            var t = dot(view.cam.dir,d)/dot(view.cam.dir,vc);

            var raw_pos = view.cam.pos+vc*t;

            var cosa = dot(view.screen.top, view.cam.dir)/
                (length(view.screen.top)*length(view.cam.dir));
            var center = view.cam.pos-d;
            var on_top = normalize((center + view.screen.top) - (view.cam.dir*cosa) - center);
            var on_right = cross(view.cam.dir, on_top);

            var vec_pos = raw_pos-center;
            cosa = dot(on_top, vec_pos)/(length(on_top)*length(vec_pos));
            ret.position.y = -length(on_top*cosa)*cosa/abs(cosa);
            cosa = dot(on_right, vec_pos)/(length(on_right)*length(vec_pos));
            ret.position.x = length(on_right*cosa)*cosa/abs(cosa);

            ret.position.z = 0.0;
            ret.position.w = 1.0;
            return ret;
        }

        @fragment fn fragment_main(fsInput: VertexOut)->@location(0) vec4f{
            return fsInput.color;
        }`
    });
    render_pipeline = device.createRenderPipeline({
        label: 'render pipeline',
        layout: 'auto',
        vertex: {
            module: render_module,
            entryPoint: "vertex_main",
        },
        fragment:{
            module: render_module,
            entryPoint: "fragment_main",
            targets:[{format: presentationFormat}]
        },
        primitive:{
            topology: 'triangle-list'
        }
    });

    renderPassDescriptor = {
        label: "render pass descriptor",
        colorAttachments: [
            {
                clearValue: [0.3, 0.1, 0.3, 1],
                loadOp: 'clear',
                storeOp: 'store'
            }
        ]
    };

    const viewSize = 11*4;
    const viewBuffer = device.createBuffer({
        size: viewSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    const view = new Float32Array([
        -1.0, -1.0, 5,             //view.campos
        0.5, 0.5, -0.4,        //view.camdir
        1.0,                            //view.screensize
        1.0,                            //view.screenindent
        0.0, 0.0, 1.0              //view.screentop
    ]);
    device.queue.writeBuffer(viewBuffer, 0, view);

    renderBindGroup = device.createBindGroup({
        label: 'render bind group',
        layout: render_pipeline.getBindGroupLayout(0),
        entries:[
        { binding: 0, resource: { buffer: workBuffer }},
        { binding: 1, resource: { buffer: viewBuffer }},
        { binding: 2, resource: { buffer: sideBuffer }}
      ]
    });
}

var t = 0;

async function full_exec(){
    device.queue.writeBuffer(timeBuffer, 0, new Float32Array([t]))
    await exec();
    await render();
    t+=0.1;
}

async function main() {
    if(!await config_workspace()){
        return;
    }
    await config_compute();
    await config_render();
    setInterval(full_exec, 10);
}


main();

function toggleGravity(check){
    if(check.checked){
        this.device.queue.writeBuffer(this.acclBuffer, 0, new Float32Array([0.098]));
    }else{
        this.device.queue.writeBuffer(this.acclBuffer, 0, new Float32Array([0]));
    }
}