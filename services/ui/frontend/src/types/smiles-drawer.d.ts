// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * smiles-drawer ships type declarations under dist/types but does not point
 * package.json at them, so TypeScript cannot find them. Its ESM build exports
 * only the namespace as a default, so that is how this declares it.
 *
 * Only the part of the API this app uses is declared.
 */
declare module "smiles-drawer" {
  interface SmiDrawer {
    /**
     * Draw *smiles* into *target*. Appends to the target rather than
     * replacing its contents, so callers must clear it before redrawing.
     */
    draw(
      smiles: string,
      target: SVGElement | HTMLCanvasElement | string,
      theme?: string,
      successCallback?: (() => void) | null,
      errorCallback?: ((error: Error) => void) | null,
    ): void;
  }

  interface SmiDrawerConstructor {
    new (
      moleculeOptions?: Record<string, unknown>,
      reactionOptions?: Record<string, unknown>,
    ): SmiDrawer;
  }

  const SmilesDrawer: {
    SmiDrawer: SmiDrawerConstructor;
    clean(smiles: string): string;
    Version: string;
  };

  export default SmilesDrawer;
}
