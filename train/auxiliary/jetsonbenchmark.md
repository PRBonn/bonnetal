# Benchmark on NVIDIA jetson

# Semantic Segmentation

People vs Background CNNs. All numbers are frames-per-second (FPS). Higher is better :)


<table class="tg">
  <tr>
    <th class="tg-c3ow" rowspan="2">Arch<br>[input size]<br>vs <br>GPU</th>
    <th class="tg-c3ow" colspan="2">MobilenetsV2 - ASPP</th>
    <th class="tg-c3ow" colspan="2">ERFNet</th>
  </tr>
  <tr>
    <td class="tg-baqh">[640x480]</td>
    <td class="tg-baqh">[320x240]</td>
    <td class="tg-baqh">[640x480]</td>
    <td class="tg-baqh">[320x240]</td>
  </tr>
  <tr>
    <td class="tg-0pky">Nano FP16</td>
    <td class="tg-0pky">2.78</td>
    <td class="tg-0pky">11.5</td>
    <td class="tg-0pky">5.5</td>
    <td class="tg-0pky">20.8</td>
  </tr>
  <tr>
    <td class="tg-0pky">TX2-fp16</td>
    <td class="tg-0pky">7</td>
    <td class="tg-0pky">28.57</td>
    <td class="tg-0pky">14.49</td>
    <td class="tg-0pky">52.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">Xavier-fp16</td>
    <td class="tg-0pky">44.5</td>
    <td class="tg-0pky">142.8</td>
    <td class="tg-0pky">62.5</td>
    <td class="tg-0pky">166.6</td>
  </tr>
  <tr>
    <td class="tg-0pky">Xavier-int8</td>
    <td class="tg-0pky">62.5</td>
    <td class="tg-0pky">166.6</td>
    <td class="tg-0pky">83.5</td>
    <td class="tg-0pky">200</td>
  </tr>
  <tr>
    <td class="tg-0lax">Xavier-fp16-DLA*</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">21.7</td>
    <td class="tg-0lax">45.5</td>
  </tr>
</table>

# Depth from ZED SDK

Info about modes [HERE](https://www.stereolabs.com/docs/depth-sensing/advanced-settings/#depth-modes). All numbers are frames-per-second (FPS). Higher is better :)

<table class="tg">
  <tr>
    <th class="tg-c3ow" colspan="2">GPU</th>
    <th class="tg-c3ow" rowspan="2">Nano</th>
    <th class="tg-c3ow" rowspan="2">TX2</th>
    <th class="tg-c3ow" rowspan="2">Xavier</th>
  </tr>
  <tr>
    <td class="tg-c3ow">Mode</td>
    <td class="tg-0pky">Resolution / RGB FPS</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">Performance</td>
    <td class="tg-0pky">VGA / 100 FPS</td>
    <td class="tg-0pky">33</td>
    <td class="tg-0pky">49</td>
    <td class="tg-0pky">95</td>
  </tr>
  <tr>
    <td class="tg-0pky">720p / 60 FPS</td>
    <td class="tg-0pky">17</td>
    <td class="tg-0pky">29</td>
    <td class="tg-0pky">45</td>
  </tr>
  <tr>
    <td class="tg-0pky">1080p / 30 FPS</td>
    <td class="tg-0pky">9</td>
    <td class="tg-0pky">16</td>
    <td class="tg-0pky">26</td>
  </tr>
  <tr>
    <td class="tg-0pky">2K / 15 FPS</td>
    <td class="tg-0pky">6.5</td>
    <td class="tg-0pky">11</td>
    <td class="tg-0pky">15</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">Medium</td>
    <td class="tg-0pky">VGA / 100 FPS</td>
    <td class="tg-0pky">12</td>
    <td class="tg-0pky">23</td>
    <td class="tg-0pky">59</td>
  </tr>
  <tr>
    <td class="tg-0pky">720p / 60 FPS</td>
    <td class="tg-0pky">9</td>
    <td class="tg-0pky">17</td>
    <td class="tg-0pky">35</td>
  </tr>
  <tr>
    <td class="tg-0pky">1080p / 30 FPS</td>
    <td class="tg-0pky">6</td>
    <td class="tg-0pky">10</td>
    <td class="tg-0pky">21</td>
  </tr>
  <tr>
    <td class="tg-0pky">2K / 15 FPS</td>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">10</td>
    <td class="tg-0pky">15</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">Quality</td>
    <td class="tg-0pky">VGA / 100 FPS</td>
    <td class="tg-0pky">6</td>
    <td class="tg-0pky">12</td>
    <td class="tg-0pky">36</td>
  </tr>
  <tr>
    <td class="tg-0pky">720p / 60 FPS</td>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">10</td>
    <td class="tg-0pky">26</td>
  </tr>
  <tr>
    <td class="tg-0pky">1080p / 30 FPS</td>
    <td class="tg-0pky">3.5</td>
    <td class="tg-0pky">7</td>
    <td class="tg-0pky">16</td>
  </tr>
  <tr>
    <td class="tg-0pky">2K / 15 FPS</td>
    <td class="tg-0pky">3.5</td>
    <td class="tg-0pky">6</td>
    <td class="tg-0pky">15</td>
  </tr>
  <tr>
    <td class="tg-0pky" rowspan="4">Ultra</td>
    <td class="tg-0pky">VGA / 100 FPS</td>
    <td class="tg-0pky">26</td>
    <td class="tg-0pky">47</td>
    <td class="tg-0pky">93</td>
  </tr>
  <tr>
    <td class="tg-0pky">720p / 60 FPS</td>
    <td class="tg-0pky">7</td>
    <td class="tg-0pky">15</td>
    <td class="tg-0pky">33</td>
  </tr>
  <tr>
    <td class="tg-0pky">1080p / 30 FPS</td>
    <td class="tg-0pky">5.5</td>
    <td class="tg-0pky">11</td>
    <td class="tg-0pky">21</td>
  </tr>
  <tr>
    <td class="tg-0pky">2K / 15 FPS</td>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">9</td>
    <td class="tg-0pky">15</td>
  </tr>
</table>