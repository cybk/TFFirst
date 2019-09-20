function createModel () {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single hidden layer
    model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    //Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));
    return model;
}

function convertToTensor (data) {
    return tf.tidy(() => {
        // Shuffle the data
        tf.util.shuffle(data);
        // Convert data to Tensor
        const inputs = data.map(d => d.rooms);
        console.log('inputs', inputs);
        const labels = data.map(d => d.price);
        console.log('labels', labels);
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        // Normalize the data to the range 0-1 using min max scaling

        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin
        }
    });
}

function testModel (model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling 
    // that we did earlier.

    const [xs, preds] = tf.tidy(() => {
        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));

        const unNormXs = xs
            .mul(inputMax.sub(inputMin))
            .add(inputMin);

        const unNormPreds = preds
            .mul(labelMax.sub(labelMin))
            .add(labelMin);
        
        // Un-normalize the data
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
        return {x: val, y: preds[i]}
      });
      
      const originalPoints = inputData.map(d => ({
        x: d.rooms, y: d.price,
      }));
      
      
      tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data'}, 
        {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
        {
          xLabel: 'No. of rooms',
          yLabel: 'Price',
          height: 300
        }
      );
}

async function trainModel (model, inputs, labels) {
    // Prepare the model for training
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 28;
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            {name: 'Training Performance'},
            ['loss', 'mse'],
            {height: 200, callbacks:['onEpochEnd']}
        )
    });
}

async function getData () {
    const houseDataReq = await fetch('https://raw.githubusercontent.com/meetnandu05/ml1/master/house.json');
    const houseData = await houseDataReq.json();
    const cleaned = houseData
    .map((item) => ({
       price: item.Price,
       rooms: item.AvgAreaNumberofRooms 
    }))
    .filter(house => (house.price != null && house.rooms != null));

    return cleaned;
}

async function run () {
    // Load and plot the original input data
    const data = await getData();
    const values = data.map(item => ({
        x: item.rooms,
        y: item.price
    }));

    tfvis.render.scatterplot(
        { name: 'No. of Rooms vs Price'},
        { values },
        {
            xLabel: 'No. of Rooms',
            yLabel: 'Price',
            heignt: 500
        }
    );

    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Convert the data to a form we can use for training
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    console.log('starting training');

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    testModel(model, data, tensorData);
}

document.addEventListener('DOMContentLoaded', run);