<template>
  <div id="Header">
    <div class="top-left-edition">
      <span style="color: #2e87cf; font-weight: bold">
        <i class="el-icon-star-off" style="font-size: 23px"></i
        >红外与可见光图像融合
      </span>
      <span>
        <i class="el-icon-time" style="font-size: 23px"></i>作者:lailihao
      </span>
    </div>
    <div id="word">
      <h1>{{ msg }}</h1>
    </div>
  </div>
</template>
<script>
export default {
  name: "Header",
  data() {
    return {
      msg: "WEB展示端",
      activeIndex: "1",
    };
  },
  methods: {},
};
</script>
<style scoped>
#Header {
  padding: 30px 110px 0 150px;
  width: 90%;
  margin: 10px auto;
  /* border-bottom: 1px solid #2e87cf; */
}

#word {
  margin-left: 45%;
  margin-top: -35px;
  margin-bottom: 37px;
  height: 60px;
  line-height: 3.2em;
  letter-spacing: 8px;
}

h1 {
  /*text-align: center;*/
  color: #2e87cf;
  letter-spacing: 30px;
  font-size: 2.3em;
}

.el-menu-demo {
  width: 80%;
  margin: 0px auto;
  padding: 0px auto;
}

.top-left-edition span i {
  float: left;
  margin-right: 10px;
}

i,
input,
label {
  vertical-align: middle;
}

i {
  border: 0;
  display: block;
  cursor: pointer;
}

.top-left-edition span {
  float: left;
  font-size: 16px;
  color: #999999;
  line-height: 24px;
  margin-right: 40px;
}
</style>


