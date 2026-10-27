function Subscript({ symbol, index, vector = false }) {
  return (
    <msub>
      <mi className={vector ? 'dev-math-vector' : undefined}>{symbol}</mi>
      <mtext>{index}</mtext>
    </msub>
  )
}

function Score({ index }) {
  return <Subscript symbol="S" index={index} />
}

function ConfidenceFunction() {
  return <mrow><mi>g</mi><mo>(</mo><mi>h</mi><mo>)</mo></mrow>
}

function CosineEquation() {
  return (
    <math display="block" aria-label="Cosine similarity between embeddings a and b">
      <mrow>
        <mi>S</mi><mo>(</mo><mi>a</mi><mo>,</mo><mi>b</mi><mo>)</mo><mo>=</mo>
        <mfrac>
          <mrow><Subscript symbol="e" index="a" vector /><mo>·</mo><Subscript symbol="e" index="b" vector /></mrow>
          <mrow><mo>‖</mo><Subscript symbol="e" index="a" vector /><mo>‖</mo><mo>·</mo><mo>‖</mo><Subscript symbol="e" index="b" vector /><mo>‖</mo></mrow>
        </mfrac>
      </mrow>
    </math>
  )
}

function ImageEquation() {
  return (
    <math display="block" aria-label="Image-only retrieval score">
      <mrow>
        <Score index="image" /><mo>=</mo><Score index="II" /><mo>+</mo>
        <Subscript symbol="λ" index="I" /><mo>·</mo><ConfidenceFunction /><mo>·</mo><mi>C</mi>
      </mrow>
    </math>
  )
}

function TextEquation() {
  return (
    <math display="block" aria-label="Text-only retrieval score">
      <mrow>
        <Score index="T" /><mo>=</mo><mi>β</mi><mo>·</mo><Score index="TI" /><mo>+</mo>
        <mo>(</mo><mn>1</mn><mo>−</mo><mi>β</mi><mo>)</mo><mo>·</mo><Score index="TT" />
      </mrow>
    </math>
  )
}

function CombinedEquation() {
  return (
    <div className="dev-equation-lines" aria-label="Combined image and text retrieval equations">
      <math display="block">
        <mrow>
          <Score index="base" /><mo>=</mo><Subscript symbol="ω" index="I" /><mo>·</mo><Score index="II" /><mo>+</mo>
          <mo>(</mo><mn>1</mn><mo>−</mo><Subscript symbol="ω" index="I" /><mo>)</mo><mo>·</mo><Score index="T" />
        </mrow>
      </math>
      <math display="block">
        <mrow>
          <Score index="attr" /><mo>=</mo><mo>(</mo><mn>1</mn><mo>−</mo><Subscript symbol="ω" index="A" /><mo>)</mo><mo>·</mo>
          <Score index="base" /><mo>+</mo><Subscript symbol="ω" index="A" /><mo>·</mo><mi>A</mi>
        </mrow>
      </math>
      <math display="block">
        <mrow>
          <Score index="final" /><mo>=</mo><Score index="attr" /><mo>+</mo><Subscript symbol="λ" index="C" /><mo>·</mo>
          <ConfidenceFunction /><mo>·</mo><mi>C</mi>
        </mrow>
      </math>
    </div>
  )
}

function CategoryEquation() {
  return (
    <div className="dev-equation-lines dev-equation-lines--category" aria-label="Category confidence equations">
      <math display="block">
        <mrow>
          <mi>h</mi><mo>=</mo>
          <mfrac>
            <Subscript symbol="r" index="1" />
            <mrow><Subscript symbol="r" index="1" /><mo>+</mo><Subscript symbol="r" index="2" /></mrow>
          </mfrac>
        </mrow>
      </math>
      <math display="block">
        <mrow>
          <ConfidenceFunction /><mo>=</mo>
          <mo fence="true" stretchy="true">{'{'}</mo>
          <mtable columnalign="center left" columnspacing="1.4em" rowspacing="0.6em">
            <mtr><mtd><mn>0</mn></mtd><mtd><mi>h</mi><mo>≤</mo><Subscript symbol="h" index="0" /></mtd></mtr>
            <mtr>
              <mtd>
                <mfrac>
                  <mrow><mi>h</mi><mo>−</mo><Subscript symbol="h" index="0" /></mrow>
                  <mrow><Subscript symbol="h" index="1" /><mo>−</mo><Subscript symbol="h" index="0" /></mrow>
                </mfrac>
              </mtd>
              <mtd><Subscript symbol="h" index="0" /><mo>&lt;</mo><mi>h</mi><mo>&lt;</mo><Subscript symbol="h" index="1" /></mtd>
            </mtr>
            <mtr><mtd><mn>1</mn></mtd><mtd><mi>h</mi><mo>≥</mo><Subscript symbol="h" index="1" /></mtd></mtr>
          </mtable>
        </mrow>
      </math>
    </div>
  )
}

export default function RankingEquation({ type }) {
  if (type === 'cosine') return <CosineEquation />
  if (type === 'image') return <ImageEquation />
  if (type === 'text') return <TextEquation />
  if (type === 'combined') return <CombinedEquation />
  if (type === 'category') return <CategoryEquation />
  return null
}
