import { useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Switch } from "@/components/ui/switch"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Toaster } from "@/components/ui/sonner"
import { toast } from "sonner"

export function App() {
  const [strict, setStrict] = useState(false)

  return (
    <div className="min-h-svh bg-background text-foreground">
      <header className="flex items-center justify-between border-b border-border px-6 py-4">
        <div className="mono-label !text-foreground">
          Trustcall / System One
        </div>
        <Badge>demo</Badge>
      </header>
      <main className="mx-auto flex max-w-2xl flex-col gap-8 px-6 py-10">
        <section className="flex flex-col gap-3">
          <p className="mono-label">Placeholder</p>
          <h1 className="text-2xl font-medium">Structured extraction, shown live.</h1>
          <p className="text-muted-foreground">
            Theme preview. Orange marks the primary action and progress only.
          </p>
          <div className="flex items-center gap-3">
            <Button onClick={() => toast("Extraction queued")}>Run extraction</Button>
            <Button variant="outline">Secondary</Button>
          </div>
          <Progress value={42} className="max-w-xs" />
        </section>

        <Tabs defaultValue="input">
          <TabsList>
            <TabsTrigger value="input">Input</TabsTrigger>
            <TabsTrigger value="result">Result</TabsTrigger>
            <TabsTrigger value="trace">Trace</TabsTrigger>
          </TabsList>
          <TabsContent value="input" className="pt-4 text-muted-foreground">
            Text goes here.
          </TabsContent>
          <TabsContent value="result" className="pt-4 font-code text-sm text-muted-foreground">
            {'{ "name": "Ada" }'}
          </TabsContent>
          <TabsContent value="trace" className="pt-4 text-muted-foreground">
            Calls go here.
          </TabsContent>
        </Tabs>

        <label className="flex items-center gap-3 text-sm">
          <Switch checked={strict} onCheckedChange={setStrict} />
          <span className="mono-label">Strict mode</span>
        </label>
      </main>
      <Toaster />
    </div>
  )
}

export default App
