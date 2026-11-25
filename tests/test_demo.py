"""Check the actual browser JS against Python over all 100 supported inputs."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import unittest
from src.intention_model import intention_score
from rain_score import score as python_score60

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('build_demo',ROOT/'demo/build_demo.py')
build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)

class DemoChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=build.payload()

    def node(self,code):
        result=subprocess.run(['node','-e',"const app=require('./demo/core.js');const d=JSON.parse(require('fs').readFileSync(0,'utf8'));"+code],
                              cwd=ROOT,input=json.dumps(self.data),capture_output=True,text=True,encoding='utf-8',check=True)
        return json.loads(result.stdout)

    def test_all_supported_inputs_match_existing_python(self):
        values=self.node("let r=[];for(const c of ['50','60'])for(const e of c==='50'?[10,60]:[0,1])for(let a=1;a<=5;a++)for(let b=1;b<=5;b++)r.push([c,e,a,b,app.score(d,c,e,a,b)]);console.log(JSON.stringify(r));")
        self.assertEqual(len(values),100)
        for c,e,a,b,value in values:
            expected=intention_score(self.data['model50'],pop=50,outdoor_minutes=e,carry_discomfort=a,wet_discomfort=b)['score'] if c=='50' else python_score60(self.data['model60'],60,e,a,b)
            self.assertLessEqual(abs(value-expected),.005001)
            if c=='50':
                m=self.data['model50'];z=m['intercept']+sum(w*(v-mean)/scale for w,v,mean,scale in zip(m['coefficients'],[e,a,b],m['mean'],m['scale']))
            else:z=self.data['model60']['intercept']+sum(w*v for w,v in zip(self.data['model60']['weights'],[e,a,b]))
            import math
            self.assertAlmostEqual(value,100/(1+math.exp(-z)),places=11)

    def test_frozen_counts_and_missing_vs_false_alarm(self):
        values=self.node("console.log(JSON.stringify([50,60,70,'always'].map(t=>app.counts(d.windows,t))));")
        self.assertEqual([(v['falseAlarm'],v['miss']) for v in values],[(2,0),(2,0),(0,2),(42,0)])
        out=self.node("console.log(JSON.stringify([app.outcome(true,1),app.outcome(true,0),app.outcome(false,1),app.outcome(false,0)]));")
        self.assertEqual(out,['강수 적중','헛경보','놓침','무강수 적중'])

    def test_invalid_inputs_rejected(self):
        checks=self.node("let n=0;for(const args of [['70',10,4,4],['50',30,4,4],['60',2,3,4],['50',10,1.5,4],['60',0,3,6]]){try{app.score(d,...args)}catch(e){n++}}console.log(JSON.stringify(n));")
        self.assertEqual(checks,5)

    def test_single_file_has_no_external_dependencies(self):
        html=(ROOT/'docs/demo/index.html').read_text(encoding='utf-8')
        self.assertNotRegex(html,r'<(?:script|link|img)[^>]+(?:src|href)=["\'](?:https?:|//)')
        self.assertNotRegex(html,r'\b(?:fetch|XMLHttpRequest|WebSocket|localStorage|sessionStorage)\s*[.(]')
        self.assertIn("connect-src 'none'",html)
        for script in re.findall(r'<script>(.*?)</script>',html,re.S):
            subprocess.run(['node','--check'],input=script,text=True,check=True,capture_output=True)

if __name__=='__main__':unittest.main()
